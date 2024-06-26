import pandas as pd
import polars as pl
import sys

from dotenv import load_dotenv
load_dotenv()

incsvpath = sys.argv[1]
incsvsep = sys.argv[2]
outcsvpath = sys.argv[3]
outdvcpath = sys.argv[4]

unitreplace = [['tCO2e', 't'],
               ['p-km', 'pkm'], ['Mkm', 'Gm'],
               ['€', 'EUR']]

# ---------------------------------------------------------------------------------------
df = pl.read_csv(incsvpath, separator = incsvsep, infer_schema_length = 1000)

droplist = ['Description']
for col in df.columns:
    if df.select(col).unique().to_series(0).to_list() == [None]:
        droplist.append(col)
df = df.drop(droplist)

context = []
values = []
for c in df.columns:
    if c.isdigit():
        values.append(c)
    else:
        context.append(c)

dims = [c for c in context if c not in ['Quantity', 'Unit']]

duplicates = df.group_by(dims).agg(pl.len()).filter(pl.col('len') > 1)
if len(duplicates) > 0:
    print('There are duplicate values. Remove them and try again.')
    print(duplicates)
    exit()
else:
    print('No duplicates, continuing...')

missing_data = df.filter((pl.col('Sector') + pl.col('Quantity') + pl.col('Unit')).is_null())
if missing_data.is_empty():
    print('No missing data in columns Sector, Quantity, Unit. Continuing...')
else:
    print('Missing data in obligatory cells. Fill in and try again.')
    print(missing_data.select(['Sector', 'Quantity', 'Unit']))
    exit()

unitcol = df.select('Unit').to_series(0).to_list()
for ur in unitreplace:
    unitcol = [x.replace(ur[0], ur[1]) for x in unitcol]
df = df.with_columns(pl.Series(name = 'Unit', values = unitcol))

if 'Scope' in df.columns:   # Detailed actions have no 'Scope' column.
    scopecol = df.select('Scope').to_series(0).to_list()
    labels = []
    for x in scopecol:
        if x:
            labels.append('Scope %i' % x)
        else:
            labels.append(x)
    df = df.with_columns(pl.Series(name = 'Scope', values = labels))

# ---------------------------------------------------------------------------------------
dfmain = df.head(1).select(context).with_columns([(pl.lit(0.0).alias('Value').cast(pl.Float64)),
                                                  (pl.lit(0).alias('Year').cast(pl.Int64))]).clear()

df = df.with_row_index(name = 'Index')
for i in range(len(df)):  # FIXME This loop is becoming increasingly slow as the length of the df increases: 2s/row 
    print('Row', i, 'out of', len(df))
    for y in values:
        mcols = list(context)
        mcols.extend([y])

        mframe = df.filter(pl.col('Index') == i).select(mcols).with_columns(pl.lit(y).cast(pl.Int64))
        mframe.columns = dfmain.columns
        if mframe['Value'][0] is not None:
            dfmain = pl.concat([dfmain, mframe])

if outcsvpath.upper() not in ['N', 'NONE']:
    dfmain.write_csv(outcsvpath)

if outdvcpath.upper() not in ['N', 'NONE']:
    from dvc_pandas import Dataset, Repository

    indexcols = list(dims)
    indexcols.extend(['Year'])
    if 'Quantity' in dfmain.columns:   # Detailed actions have no 'Quantity' column.
        indexcols.extend(['Quantity'])
    pdindex = pd.MultiIndex.from_frame(pd.DataFrame(dfmain.select(indexcols).fill_null('.'), columns = indexcols))

    valuecols = list(set(dfmain.columns) - set(indexcols))
    pdframe = pd.DataFrame(dfmain.select(valuecols), index = pdindex, columns = valuecols)

    ds = Dataset(pdframe, identifier = outdvcpath)
    repo = Repository(repo_url = 'git@github.com:kausaltech/dvctest.git', dvc_remote = 'kausal-s3')
    repo.push_dataset(ds)