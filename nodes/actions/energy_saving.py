import pandas as pd
import numpy as np
from numba import njit, int32
from pint_pandas import PintType

from common.i18n import gettext_lazy as _
from nodes import NodeDimension
from nodes.constants import ENERGY_QUANTITY, CURRENCY_QUANTITY, FORECAST_COLUMN, VALUE_COLUMN
from nodes.calc import nafill_all_forecast_years
from params import NumberParameter

from .action import ActionNode

from nodes.costs import DISCOUNT_RATE, HEALTH_IMPACTS_PER_KWH, AVOIDED_ELECTRICITY_CAPACITY_PRICE, HEAT_CO2_EF, ELECTRICITY_CO2_EF, COST_CO2, TIMESPAN


@njit(cache=True)
def simulate_led_retrofit(
    nr_trad: int, nr_led: int, nr_changed_per_year: int, nr_yearly_increase: int, nr_years: int
):
    trad = np.empty(nr_years, int32)
    led = np.empty(nr_years, int32)
    nr_new_led = np.empty(nr_years, int32)
    for year in range(nr_years):
        change = min(nr_changed_per_year, nr_trad)
        nr_trad -= change
        nr_led += change
        # Assume that the increase in total number of luminaires is
        # only LEDs.
        nr_led += nr_yearly_increase
        nr_new_led[year] = change + nr_yearly_increase
        trad[year] = nr_trad
        led[year] = nr_led
    return nr_new_led, trad, led


class LEDRetrofitAction(ActionNode):
    dimensions = {
        ENERGY_QUANTITY: NodeDimension('MWh/a', ENERGY_QUANTITY),
        CURRENCY_QUANTITY: NodeDimension('EUR/a', CURRENCY_QUANTITY),
    }
    allowed_parameters = [
        NumberParameter(
            local_id='yearly_retrofit_number_baseline',
            label=_('Number of LED bulbs changed per year (baseline)'),
            unit='pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='yearly_retrofit_number',
            label=_('Number of additional LED bulbs changed per year'),
            unit='pcs/a',
        ),
        NumberParameter(
            local_id='yearly_demand_increase',
            label=_('Yearly increase in total number of luminaires'),
            unit='pcs/a',
        ),
        NumberParameter(
            local_id='traditional_luminaire_maintenance_cost',
            label=_('Yearly maintenance cost of traditional luminaires'),
            unit='EUR/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_maintenance_cost',
            label=_('Yearly maintenance cost of LED luminaires'),
            unit='EUR/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='traditional_luminaire_power',
            label=_('Traditional luminaire power consumption'),
            unit='W',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_power',
            label=_('LED luminaire power consumption'),
            unit='W',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='traditional_luminaire_active_time',
            label=_('Traditional luminaire yearly active time'),
            unit='h/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_active_time',
            label=_('LED luminaire yearly active time'),
            unit='h/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_maintenance_cost',
            label=_('Yearly maintenance cost of LED luminaires'),
            unit='EUR/pcs/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='led_luminaire_investment_cost',
            label=_('Investment cost of one LED retrofit'),
            unit='EUR/pcs',
            is_customizable=False,
        ),
    ]

    def compute_effect(self) -> pd.DataFrame:
        # Input time series are:
        #  - number of luminaires with traditional (high-power) bulbs, historical
        #  - number of luminaires with LED bulbs, historical
        #  - price of electricity, historical + forecast

        trad_df = self.get_input_node(tag='traditional').get_output()
        led_df = self.get_input_node(tag='led').get_output()
        el_price = self.get_input_node(tag='price_of_electricity').get_output()
        target_year = self.get_target_year()

        df = nafill_all_forecast_years(trad_df, target_year)
        df['NrLED'] = nafill_all_forecast_years(led_df, target_year)[VALUE_COLUMN]
        df = df.rename(columns={VALUE_COLUMN: 'NrTraditional'})

        last_hist_year = df.loc[~df[FORECAST_COLUMN]].index.max()
        yearly_baseline_change = self.get_parameter_value('yearly_retrofit_number_baseline', required=False)
        if yearly_baseline_change is None:
            yearly_baseline_change = 0
        yearly_change = self.get_parameter_value('yearly_retrofit_number')
        if not self.is_enabled():
            # If the action is disabled, we assume that only the baseline amount
            # of retrofits are done.
            yearly_change = yearly_baseline_change

        nr_trad = int(self.strip_units(df['NrTraditional']).loc[last_hist_year])
        nr_led = int(self.strip_units(df['NrLED']).loc[last_hist_year])
        el_price = el_price[VALUE_COLUMN]

        # Predict the number of:
        #  - new LED luminaires installed (retrofits + yearly increase)
        #  - traditional luminaires left
        #  - LED luminaires
        nr_new_led, trad, led = simulate_led_retrofit(
            nr_trad=nr_trad,
            nr_led=nr_led,
            nr_changed_per_year=yearly_baseline_change + yearly_change,
            nr_yearly_increase=self.get_parameter_value('yearly_demand_increase'),
            nr_years=target_year - last_hist_year,
        )
        df.loc[df.index > last_hist_year, 'NrTraditional'] = trad
        df.loc[df.index > last_hist_year, 'NrLED'] = led
        df.loc[df.index > last_hist_year, 'NrNewLED'] = nr_new_led
        df['NrNewLED'] = df['NrNewLED'].astype('pint[pcs/a]')

        # Calculate energy consumption, energy cost and maintenance cost
        # for traditional luminaires
        active_time = self.get_parameter_value_w_unit('traditional_luminaire_active_time')
        power = self.get_parameter_value_w_unit('traditional_luminaire_power')
        maint_cost = self.get_parameter_value_w_unit('traditional_luminaire_maintenance_cost')
        df['TraditionalEnergy'] = df['NrTraditional'] * active_time * power
        df['TraditionalEnergy'] = df['NrTraditional'] * active_time * power
        df['TraditionalEnergyCost'] = (df['TraditionalEnergy'] * el_price).astype('pint[EUR/a]')
        df['TraditionalMaintenanceCost'] = (df['NrTraditional'] * maint_cost).astype('pint[EUR/a]')

        # Ditto for LEDs, but include yearly investment costs
        active_time = self.get_parameter_value_w_unit('led_luminaire_active_time')
        power = self.get_parameter_value_w_unit('led_luminaire_power')
        maint_cost = self.get_parameter_value_w_unit('led_luminaire_maintenance_cost')
        inv_cost = self.get_parameter_value_w_unit('led_luminaire_investment_cost')
        df['LEDEnergy'] = df['NrLED'] * active_time * power
        df['LEDEnergyCost'] = (df['LEDEnergy'] * el_price).astype('pint[EUR/a]')
        df['LEDMaintenanceCost'] = (df['NrLED'] * maint_cost).astype('pint[EUR/a]')
        df['LEDInvestmentCost'] = (df['NrNewLED'] * inv_cost)

        total_cost = (
            df['TraditionalEnergyCost'] + df['TraditionalMaintenanceCost']
            + df['LEDEnergyCost'] + df['LEDMaintenanceCost'] + df['LEDInvestmentCost']
        )
        energy_consumption = df['TraditionalEnergy'] + df['LEDEnergy']
        df[CURRENCY_QUANTITY] = total_cost.astype(PintType(self.dimensions[CURRENCY_QUANTITY].unit))
        df[ENERGY_QUANTITY] = energy_consumption.astype(PintType(self.dimensions[ENERGY_QUANTITY].unit))
        df = df[[CURRENCY_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN]]
        return df

############################################


class BuildingEnergySavingAction(ActionNode):
    dimensions = {
        ENERGY_QUANTITY: NodeDimension('kWh/a', ENERGY_QUANTITY),
        CURRENCY_QUANTITY: NodeDimension('EUR/kWh', CURRENCY_QUANTITY),
    }
    allowed_parameters = [
        NumberParameter(
            local_id='investment_lifetime',
            label=_('Investment lifetime (a)'),
            unit='a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='investment_cost',
            label=_('Investment cost (EUR/m2)'),
            unit='EUR/m**2',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='maintenance_cost',
            label=_('Maintenance cost (EUR/m2/a)'),
            unit='EUR/m**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='heat_saving',
            label=_('Heat saving (kWh/m2/a'),
            unit='kWh/m**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='electricity_saving',
            label=_('Electricity saving (kWh/m2/a)'),
            unit='kWh/m**2/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='renovation_potential',
            label=_('Renovation potential (% of floor area)'),
            unit='%',
        ),
        NumberParameter(
            local_id='renovation_rate_baseline',
            label=_('Renovation rate baseline (% of floor area / a)'),
            unit='%/a',
            is_customizable=False,
        ),
        NumberParameter(
            local_id='renovation_rate',
            label=_('Renovation rate (% of floor area / a)'),
            unit='%',
        ),
    ]

    def compute_effect(self) -> pd.DataFrame:
        # Input time series are:
        #  - building_floor_area
        #  - electricity_price
        #  - heat_price

        def serialise(df, x):
            out = pd.Series([x.m] * len(df), index=df.index, dtype='pint[' + str(x.units) + ']')
            return(out)

        floor_df = self.get_input_node(tag='floor_area').get_output()
        he_price = self.get_input_node(tag='price_of_heat').get_output()
        el_price = self.get_input_node(tag='price_of_electricity').get_output()
        target_year = self.get_target_year()

        df = nafill_all_forecast_years(floor_df, target_year)
        df['HePrice'] = he_price[VALUE_COLUMN]
        df['ElPrice'] = el_price[VALUE_COLUMN]
        df = df.rename(columns={VALUE_COLUMN: 'FloorArea'})

        last_hist_year = df.loc[~df[FORECAST_COLUMN]].index.max()
        renovation_potential = self.get_parameter_value_w_unit('renovation_potential')
        df['RenoPot'] = serialise(df, renovation_potential)
        renovation_rate_baseline = self.get_parameter_value_w_unit('renovation_rate_baseline', required=False)
        if renovation_rate_baseline is None:
            renovation_rate_baseline = 0
        renovation_rate = self.get_parameter_value_w_unit('renovation_rate')
        df['RenoRate'] = serialise(df, renovation_rate)
        if not self.is_enabled():
            # If the action is disabled, we assume that only the baseline amount
            # of retrofits are done.
            yearly_change = renovation_rate_baseline

        # Calculate energy consumption, energy cost and maintenance cost
        lifetime = self.get_parameter_value_w_unit('investment_lifetime')
        investment_cost = self.get_parameter_value_w_unit('investment_cost') / lifetime
        df['Invest'] = serialise(df, investment_cost)
        maint_cost = self.get_parameter_value_w_unit('maintenance_cost') * lifetime
        he_saving = self.get_parameter_value_w_unit('heat_saving')
        df['HeSaving'] = serialise(df, he_saving)
        el_saving = self.get_parameter_value_w_unit('electricity_saving')
        df['ElSaving'] = serialise(df, el_saving)

        df['EnSaving'] = df['HeSaving'] + df['ElSaving']
        df['CostSaving'] = (df['ElPrice'] * el_saving + df['HePrice'] * he_saving)
        net_present_value = (1 - (1 / (1 + DISCOUNT_RATE))**TIMESPAN) / (1 - (1 / (1 + DISCOUNT_RATE)))
        df['CostSaving'] = df['CostSaving'] * net_present_value
        df['PrivateProfit'] = (df['CostSaving'] - df['Invest'])
        df['ElAvoided'] = df['ElSaving'] * AVOIDED_ELECTRICITY_CAPACITY_PRICE
        df['CostCO2'] = ((df['HeSaving'] * HEAT_CO2_EF + df['ElSaving'] * ELECTRICITY_CO2_EF) * COST_CO2).astype('pint[EUR/a/m**2]')
        df['Health'] = df['EnSaving'] * HEALTH_IMPACTS_PER_KWH
        df['SocialProfit'] = (df['ElAvoided'] + df['CostCO2'] + df['Health']) * net_present_value + df['PrivateProfit']
        social_cost_efficiency = df['SocialProfit'] / df['EnSaving'] * -1
        total_reduction = (df['EnSaving'] * df['FloorArea'] * df['RenoRate'] * df['RenoPot'])

        df[CURRENCY_QUANTITY] = social_cost_efficiency.astype(PintType(self.dimensions[CURRENCY_QUANTITY].unit))
        df[ENERGY_QUANTITY] = total_reduction.astype(PintType(self.dimensions[ENERGY_QUANTITY].unit))
        self.print_pint_df(df)
        df = df[[CURRENCY_QUANTITY, ENERGY_QUANTITY, FORECAST_COLUMN]]
        return df

        # Palauta Details
        # Lisää kokonaiskustannus
        # Selvitä miksi tulos ei poikkea BAUsta
        # Laske diskontto vasta aikasarjasta
        # Tee uusia toimenpidesolmuja
        # Tee kunnon aikasarja korjausten nopeudesta
        # Lisää toimiva käyttökustannus
        # (Tee koodi toimenpidesolmujen tekemiseen)
        # (Lataa excel-datat käyttökelpoisessa muodossa dvc:hen)
