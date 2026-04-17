import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _(a):
    import marimo as mo
    print('test' + ' ' +a)
    return


@app.cell
def _():
    a = 'halo'
    return (a,)


if __name__ == "__main__":
    app.run()
