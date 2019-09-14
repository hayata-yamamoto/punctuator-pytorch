import click


@click.group()
def cmd():
    pass


@cmd.command()
def hello_world():
    click.echo('Hello World.')
