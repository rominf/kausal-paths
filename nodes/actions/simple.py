from cmath import nan
from typing import ClassVar, Iterable


import pandas as pd
import pint_pandas
import polars as pl
from common import polars as ppl


from params.param import Parameter
from params import PercentageParameter, NumberParameter, StringParameter
from nodes.constants import FORECAST_COLUMN, NODE_COLUMN, VALUE_COLUMN, YEAR_COLUMN
from nodes.node import NodeError
from .action import ActionNode



class AdditiveAction(ActionNode):
    """Simple action that produces an additive change to a value."""
    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        if not self.is_enabled():
            df[VALUE_COLUMN] = 0.0
            df[VALUE_COLUMN] = self.ensure_output_unit(df[VALUE_COLUMN])
        return df


class CumulativeAdditiveAction(ActionNode):  # FIXME Update to deal with old-fashioned multi-metric nodes such as Tampere/private_building_energy_renovation
    """Additive action where the effect is cumulative and remains in the future."""

    allowed_parameters: ClassVar[list[Parameter]] = [
        PercentageParameter('target_year_ratio', min_value=0),
    ]

    def add_cumulatively(self, df):
        end_year = self.get_end_year()
        df = df.reindex(range(df.index.min(), end_year + 1))
        df[FORECAST_COLUMN] = True

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue

            val = df[col]
            if hasattr(val, 'pint'):
                val = val.pint.m
            val = val.fillna(0).cumsum()

            target_year_ratio = self.get_parameter_value('target_year_ratio', required=False)
            if target_year_ratio is not None:
                val *= target_year_ratio / 100

            df[col] = val
            if not self.is_enabled():
                df[col] = 0.0
            df[col] = self.ensure_output_unit(df[col])

        return df

    def compute_effect(self):
        df = self.get_input_dataset()
        return self.add_cumulatively(df)


class LinearCumulativeAdditiveAction(CumulativeAdditiveAction):
    allowed_parameters = CumulativeAdditiveAction.allowed_parameters + [
        NumberParameter('target_year_level'),
        NumberParameter(
            local_id='action_delay',
            label='Years of delay (a)',
        ),
        NumberParameter('multiplier'),
    ]

    """Cumulative additive action where a yearly target is set and the effect is linear.
    This can be modified with these parameters:
    target_year_level is the value to be reached at the target year.
    action_delay is the year when the implementation of the action starts.
    multiplier scales the size of the impact (useful between scenarios).
    """
    def compute_effect(self):
        df = self.get_input_dataset()
        start_year = df.index.min()
        delay = self.get_parameter_value('action_delay', required=False)
        if delay is not None:
            start_year = start_year + int(delay)
        target_year = self.get_target_year()
        df = df.reindex(range(start_year, target_year + 1))
        df[FORECAST_COLUMN] = True

        target_year_level = self.get_parameter_value('target_year_level', required=False)
        if target_year_level is not None:
            if set(df.columns) != set([VALUE_COLUMN, FORECAST_COLUMN]):
                raise NodeError(self, "target_year_level parameter can only be used with single-value nodes")
            df.loc[target_year, VALUE_COLUMN] = target_year_level
            if delay is not None:
                df.loc[range(start_year + 1, target_year), VALUE_COLUMN] = nan

        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            dt = df.dtypes[col]
            df[col] = df[col].pint.m.interpolate(method='linear').diff().fillna(0).astype(dt)

        df = self.add_cumulatively(df)
        for col in df.columns:
            if col == FORECAST_COLUMN:
                continue
            multiplier = self.get_parameter_value('multiplier', required=False, units=True)
            if multiplier is not None:
                df[col] *= multiplier
            df[col] = self.ensure_output_unit(df[col])
        return df


class EmissionReductionAction(ActionNode):
    """Simple emission reduction impact"""

    no_effect_value = 0

    def compute_effect(self):
        df = self.get_input_dataset()
        df[VALUE_COLUMN] = 0 - df[VALUE_COLUMN]
        return df


class ScenarioAction(ActionNode):
    '''
    First like ActionNode, but then selecting a scenario based on a parameter.
    The parameter must be given, and the df must have dimension scenario.
    '''
    allowed_parameters = ActionNode.allowed_parameters + [
        StringParameter(local_id='scenario')
    ]
    def compute_effect(self):
        df = self.get_input_dataset_pl()
        scen_id = self.get_parameter_value('scenario', required=True)
        if not self.is_enabled():
            scen_id = 'baseline'
        df = df.filter(pl.col('scenario').eq(scen_id)).drop('scenario')
        return df


class TrajectoryAction(ActionNode):
    '''
    TrajectoryAction is an ActionNode where you define the effect as an absolute trajectory of values, not as a relative change from the baseline like usually. The trajectory is converted to baseline-relative values by giving the baseline value and baseline year as parameter values. This is a bit cumbersome, but we cannot get the baseline value from the output node because that would make the graph cyclic.
    '''
    allowed_parameters = ActionNode.allowed_parameters + [
        StringParameter(local_id='scenario'),
        NumberParameter(local_id='baseline_year_level'),
        NumberParameter(local_id='baseline_year')
    ]
    def compute_effect(self):
        df = self.get_input_dataset_pl()
        scen_id = self.get_parameter_value('scenario', required=True)
        if not self.is_enabled():
            scen_id = 'baseline'
        df = df.filter(pl.col('scenario').eq(scen_id)).drop('scenario')

        level = self.get_parameter_value('baseline_year_level', required=False)
        year = int(self.get_parameter_value('baseline_year', required=True))
        df = df.filter(pl.col(YEAR_COLUMN).ge(year))
        if level is None:  # Assume a relative change
            level = df.filter(pl.col(YEAR_COLUMN).eq(year))[VALUE_COLUMN][0]
            df = df.with_columns((
                pl.col(VALUE_COLUMN) / pl.lit(level) - pl.lit(1)
            ))
            df = df.clear_unit(VALUE_COLUMN)
            df = df.set_unit(VALUE_COLUMN, 'dimensionless')
            df = df.ensure_unit(VALUE_COLUMN, self.unit)
        else:
            df = df.with_columns(pl.col(VALUE_COLUMN) - pl.lit(level))  # FIXME Check units

        return df
