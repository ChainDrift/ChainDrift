ARG sourceimage=chaindriftorg/chaindrift
ARG sourcetag=develop
FROM ${sourceimage}:${sourcetag}

# Install dependencies
COPY requirements-plot.txt /chaindrift/

RUN pip install -r requirements-plot.txt --user --no-cache-dir
