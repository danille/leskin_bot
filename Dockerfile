FROM eu.gcr.io/google-appengine/python
RUN virtualenv -p python3.7 /env

# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
ENV MODEL_NAME "full_skin_cancer_model.h5"

# Install some depedencies for openCV
RUN  apt update && apt install -y libsm6 libxext6 libxrender1

# Copy the application's requirements.txt and run pip to install all
# dependencies into the virtualenv.
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Add the application source code.
ADD . /app

EXPOSE 8080
# Run a WSGI server to serve the application. gunicorn must be declared as
# a dependency in requirements.txt.
CMD gunicorn main:app -b :8080 --timeout 600
