# Generated by Django 3.2.4 on 2021-10-22 16:14

from django.db import migrations, models


def migrate_data(apps, schema_editor):
    instance_configs = {}
    InstanceContent = apps.get_model('pages', 'InstanceContent')
    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')
    for content in InstanceContent.objects.all():
        config = InstanceConfig.objects.create(
            identifier=content.identifier,
            lead_title=content.lead_title,
            lead_paragraph=content.lead_paragraph,
        )
        instance_configs[config.identifier] = config
    NodeContent = apps.get_model('pages', 'NodeContent')
    NodeConfig = apps.get_model('nodes', 'NodeConfig')
    for content in NodeContent.objects.all():
        NodeConfig.objects.create(
            instance=instance_configs[content.instance.identifier],
            identifier=content.node_id,
            short_description=content.short_description,
            body=content.body,
        )


def migrate_data_reverse(apps, schema_editor):
    InstanceContent = apps.get_model('pages', 'InstanceContent')
    InstanceConfig = apps.get_model('nodes', 'InstanceConfig')
    for config in InstanceConfig.objects.all():
        content = InstanceContent.objects.get(identifier=config.identifier)
        content.lead_title = config.lead_title
        content.lead_paragraph = config.lead_paragraph
        content.save()
    NodeContent = apps.get_model('pages', 'NodeContent')
    NodeConfig = apps.get_model('nodes', 'NodeConfig')
    for config in NodeConfig.objects.all():
        content = NodeContent.objects.get(node_id=config.identifier)
        content.short_description = config.short_description
        content.body = config.body
        content.save()


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0004_add_lead_content'),
        ('nodes', '0002_add_fields_from_pages_app'),
    ]

    operations = [
        migrations.RunPython(migrate_data, migrate_data_reverse),
        migrations.RemoveField(
            model_name='instancecontent',
            name='lead_title',
        ),
        migrations.RemoveField(
            model_name='instancecontent',
            name='lead_paragraph',
        ),
        migrations.RemoveField(
            model_name='nodecontent',
            name='body',
        ),
        migrations.RemoveField(
            model_name='nodecontent',
            name='short_description',
        ),
        migrations.AlterField(
            model_name='instancecontent',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
        migrations.AlterField(
            model_name='nodecontent',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
    ]
