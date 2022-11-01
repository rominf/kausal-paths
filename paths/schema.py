import graphene
from graphql.type.definition import GraphQLArgument, GraphQLNonNull
from graphql.type.directives import (
    DirectiveLocation, GraphQLDirective, specified_directives
)
from graphql.type.scalars import GraphQLID, GraphQLString
from django.utils.translation import get_language

import pint
from grapple.registry import registry as grapple_registry

from nodes.schema import Query as NodesQuery
from pages.schema import Query as PagesQuery
from params.schema import (
    Mutations as ParamsMutations, Query as ParamsQuery, types as params_types
)

CO2E = 'CO<sub>2</sub>e'


class UnitType(graphene.ObjectType):
    short = graphene.String()
    long = graphene.String()
    html_short = graphene.String()
    html_long = graphene.String()

    def resolve_short(self: pint.Unit, info):  # type: ignore
        lang = get_language()
        val = self.format_babel('~P', locale=lang)
        if val == 't/a/cap':
            if lang == 'de':
                return 't/a/Einw.'
            elif lang == 'en':
                return 't/a/inh.'
        return val

    def resolve_long(self: pint.Unit, info):  # type: ignore
        lang = get_language()
        val = self.format_babel('~P', locale=lang)
        if val == 't/a/cap':
            if lang == 'de':
                return 't CO₂e/Jahr/Einw.'
            elif lang == 'en':
                return 't CO₂e/year/inh.'
        return val

    def resolve_html_short(self: pint.Unit, info):  # type: ignore
        lang = get_language()
        val = self.format_babel('~H', locale=lang)
        # FIXME: remove this hack
        if val == 't/(a cap)':
            if lang == 'de':
                return 't∕a∕Einw.'
        return val

    def resolve_html_long(self: pint.Unit, info):  # type: ignore
        lang = get_language()
        val = self.format_babel('~H', locale=lang)
        if val == 't/(a cap)':
            if lang == 'de':
                return f't {CO2E}∕Jahr∕Einw.'
            elif lang == 'en':
                return f't {CO2E}∕year∕inh.'
        elif val == 'kt/a':
            if lang == 'fi':
                return f'kt {CO2E}∕vuosi'
            elif lang == 'en':
                return f'kt {CO2E}∕year'
            elif lang == 'de':
                return f'kt {CO2E}∕Jahr'
        return val


class Query(NodesQuery, ParamsQuery, PagesQuery):
    pass


class Mutations(ParamsMutations):
    pass


class LocaleDirective(GraphQLDirective):
    def __init__(self):
        super().__init__(
            name='locale',
            description='Select locale in which to return data',
            args={
                'lang': GraphQLArgument(
                    type_=GraphQLNonNull(GraphQLString),
                    description='Selected language'
                )
            },
            locations=[DirectiveLocation.QUERY]
        )


class InstanceDirective(GraphQLDirective):
    def __init__(self):
        super().__init__(
            name='instance',
            description='Select the Paths instance for the request',
            args={
                'hostname': GraphQLArgument(
                    type_=GraphQLString,
                    description='Hostname'
                ),
                'identifier': GraphQLArgument(
                    type_=GraphQLID,
                    description='Instance identifier'
                ),
                'token': GraphQLArgument(
                    type_=GraphQLString,
                    description='Token for accessing the instance'
                ),
            },
            locations=[DirectiveLocation.QUERY, DirectiveLocation.MUTATION]
        )


schema = graphene.Schema(
    query=Query,
    directives=list(specified_directives) + [LocaleDirective(), InstanceDirective()],
    types=params_types + list(grapple_registry.models.values()),
    mutation=Mutations,
)
