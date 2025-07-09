import typer

# CLI app
app = typer.Typer(help="Model Converter Tool - Professional, Clean CLI")

# 子命令注册
from model_converter_tool.commands import inspect, convert, list_cmd, validate, cache, history, config, version

app.command()(inspect.inspect)
app.command()(convert.convert)
app.command(name="list")(list_cmd.list)
app.command()(validate.validate)
app.command()(cache.cache)
app.command()(history.history)
app.command()(config.config)
app.command()(version.version)

if __name__ == "__main__":
    app() 