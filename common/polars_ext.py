from __future__ import annotations

from typing import Literal, TypeAlias, TYPE_CHECKING, Union

from polars.datatypes import Float32, Float64
import pandas as pd
from pint_pandas import PintType

import polars as pl
import common.polars as ppl
from nodes.units import Unit
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN, FORECAST_COLUMN


if TYPE_CHECKING:
    from nodes.dimensions import Dimension
    from nodes.node import NodeMetric


Dimensions: TypeAlias = dict[str, 'Dimension']
Metrics: TypeAlias = dict[str, 'NodeMetric']

DF: TypeAlias = Union[ppl.PathsDataFrame, pl.DataFrame]


@pl.api.register_dataframe_namespace('paths')
class PathsExt:
    _df: ppl.PathsDataFrame

    def __init__(self, df: DF) -> None:
        if not isinstance(df, ppl.PathsDataFrame):
            df = ppl.to_ppdf(df)
        self._df = df

    def to_pandas(self, meta: ppl.DataFrameMeta | None = None) -> pd.DataFrame:
        return self._df.to_pandas(meta=meta)

    def to_wide(self, meta: ppl.DataFrameMeta | None = None) -> ppl.PathsDataFrame:
        """Project the DataFrame wide (dimension categories become columns) and group by year."""

        df = self._df

        if meta is None:
            meta = df.get_meta()
        dim_ids = meta.dim_ids
        metric_cols = list(meta.units.keys())
        if not metric_cols:
            metric_cols = [VALUE_COLUMN]
        for col in dim_ids + metric_cols:
            if col not in df.columns:
                raise Exception("Column %s from metadata is not present in DF")

        # Create a column '_dims' with all the categories included
        if not dim_ids:
            return df

        df = df.with_columns([
            pl.concat_list([
                pl.format('{}:{}', pl.lit(dim), pl.col(dim)) for dim in dim_ids 
            ]).arr.join('/').alias('_dims')
        ])
        mdf = None
        units = {}
        for metric_col in metric_cols:
            tdf = df.pivot(index=[YEAR_COLUMN, FORECAST_COLUMN], columns='_dims', values=metric_col)
            cols = [col for col in tdf.columns if col not in (YEAR_COLUMN, FORECAST_COLUMN)]
            metric_unit = meta.units.get(metric_col)
            if metric_unit is not None:
                for col in cols:
                    units['%s@%s' % (metric_col, col)] = metric_unit
            tdf = ppl.to_ppdf(tdf.rename({col: '%s@%s' % (metric_col, col) for col in cols}))
            if mdf is None:
                mdf = tdf
            else:
                tdf = tdf.drop(columns=FORECAST_COLUMN)
                mdf = mdf.join(tdf, on=YEAR_COLUMN)
        assert mdf is not None
        return ppl.PathsDataFrame._from_pydf(
            mdf._df,
            meta=ppl.DataFrameMeta(units=units, primary_keys=[YEAR_COLUMN])
        )

    def to_narrow(self) -> ppl.PathsDataFrame:
        df: ppl.PathsDataFrame | pl.DataFrame = self._df
        widened_cols = [col for col in df.columns if '@' in col]
        if not len(widened_cols):
            return df  # type: ignore
        tdf = df.melt(id_vars=[YEAR_COLUMN, FORECAST_COLUMN]).with_columns([
            pl.col('variable').str.split('@').alias('_tmp')
        ]).with_columns([
            pl.col('_tmp').arr.first().alias('Metric'),
            pl.col('_tmp').arr.last().str.split('/').alias('_dims'),
        ])
        df = ppl.to_ppdf(tdf)
        first = df['_dims'][0]
        dim_ids = [x.split(':')[0] for x in first]
        dim_cols = [pl.col('_dims').arr.get(idx).str.split(':').arr.get(1).alias(col) for idx, col in enumerate(dim_ids)]
        df = df.with_columns(dim_cols)
        df = df.pivot(values='value', index=[YEAR_COLUMN, FORECAST_COLUMN, *dim_ids], columns='Metric')
        df = df.with_columns([pl.col(dim).cast(pl.Categorical) for dim in dim_ids])
        return ppl.to_ppdf(df)

    def make_forecast_rows(self, end_year: int) -> ppl.PathsDataFrame:
        df: DF = self._df
        if isinstance(df, ppl.PathsDataFrame):
            meta = df.get_meta()
        else:
            meta = None
        y = df[YEAR_COLUMN]
        if y.n_unique() != len(y):
            raise Exception("DataFrame has duplicated years")

        if FORECAST_COLUMN not in df.columns:
            last_hist_year = y.max()
        else:
            last_hist_year = df.filter(~pl.col(FORECAST_COLUMN))[YEAR_COLUMN].max()
        assert isinstance(last_hist_year, int)
        years = pl.DataFrame(data=range(last_hist_year + 1, end_year + 1), schema=[YEAR_COLUMN])
        df = df.join(years, on=YEAR_COLUMN, how='outer').sort(YEAR_COLUMN)
        df = df.with_columns([
            pl.when(pl.col(YEAR_COLUMN) > last_hist_year).then(pl.lit(True)).otherwise(pl.col(FORECAST_COLUMN)).alias(FORECAST_COLUMN)
        ])
        return ppl.to_ppdf(df, meta=meta)

    def nafill_pad(self) -> ppl.PathsDataFrame:
        """Fill N/A values by propagating the last valid observation forward.

        Requires a DF in wide format (indexed by year).
        """

        df = self._df
        y = df[YEAR_COLUMN]
        if y.n_unique() != len(y):
            raise Exception("DataFrame has duplicated years")

        df = df.fill_null(strategy='forward')
        return df

    def sum_over_dims(self, dims: list[str] | None = None) -> ppl.PathsDataFrame:
        df = self._df
        meta = df.get_meta()
        if FORECAST_COLUMN in df.columns:
            fc = [pl.first(FORECAST_COLUMN)]
        else:
            fc = []

        if dims is None:
            dims = meta.dim_ids
        remaining_keys = list(meta.primary_keys)
        for dim in dims:
            remaining_keys.remove(dim)

        zdf = df.groupby(remaining_keys).agg([
            *[pl.sum(col).alias(col) for col in meta.metric_cols],
            *fc,
        ]).sort(YEAR_COLUMN)
        return ppl.to_ppdf(zdf, meta=meta)

    def join_over_index(self, other: ppl.PathsDataFrame, how: Literal['left', 'outer'] = 'left'):
        sdf = self._df
        sm = sdf.get_meta()
        om = other.get_meta()
        df = sdf.join(other, on=sm.primary_keys, how=how)
        fc_right = '%s_right' % FORECAST_COLUMN
        if FORECAST_COLUMN in df.columns and fc_right in df.columns:
            df = df.with_columns([
                pl.col(FORECAST_COLUMN).fill_null(False) | pl.col(fc_right).fill_null(False)
            ])
            df = df.drop(fc_right)
        for col in om.metric_cols:
            col_right = '%s_right' % col
            if col_right in df.columns:
                sm.units[col_right] = om.units[col]
            elif col in df.columns:
                sm.units[col] = om.units[col]

        out = ppl.to_ppdf(df, meta=sm)
        return out

    def index_has_duplicates(self) -> bool:
        df = self._df
        if not df._primary_keys:
            return False
        ldf = df.lazy()
        dupes = ldf.groupby(df._primary_keys).agg(pl.count()).filter(pl.col('count') > 1).limit(1).collect()
        return len(dupes) > 0

    def add_with_dims(self, odf: ppl.PathsDataFrame, how: Literal['left', 'outer'] = 'left') -> ppl.PathsDataFrame:
        df = self._df
        val_col = VALUE_COLUMN

        output_unit = df.get_unit(val_col)
        meta = df.get_meta()
        cols = df.columns
        odf = odf.ensure_unit(val_col, output_unit).select([YEAR_COLUMN, *meta.dim_ids, val_col, FORECAST_COLUMN])

        for dim in meta.dim_ids:
            dt = df[dim].dtype
            if odf[dim].dtype != dt:
                odf = odf.with_columns([pl.col(dim).cast(df[dim].dtype)])

        left_fc = pl.col(FORECAST_COLUMN)
        right_fc = pl.col(FORECAST_COLUMN + '_right')
        left_val = pl.col(val_col)
        right_val = pl.col(val_col + '_right')
        if how == 'outer':
            left_fc = left_fc.fill_null(False)
            right_fc = right_fc.fill_null(False)

            left_val = left_val.fill_null(0)
            right_val = right_val.fill_null(0)
        elif how == 'left':
            right_fc = right_fc.fill_null(False)
            right_val = right_val.fill_null(False)

        df = ppl.to_ppdf(df.join(odf, on=[YEAR_COLUMN, *meta.dim_ids], how=how), meta=meta)
        df = df.with_columns([
            left_val + right_val,
            left_fc | right_fc
        ])
        df = df.select(cols)
        return df
