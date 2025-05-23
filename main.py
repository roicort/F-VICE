from radicli import Radicli, Arg

cli = Radicli()

@cli.command(
    "app",
    port=Arg(
        "--port",
        help="Port to run the Streamlit app on."
    ),
    host=Arg(
        "--host",
        help="Host to run the Streamlit app on."
    ),
)
def app(port=8888, host="localhost"):
    """
    Run the Streamlit app.
    """
    import os
    os.system(f"streamlit run app.py --server.port {port} --server.address {host}")

if __name__ == "__main__":
    cli.run()