#FROM kaldi-align
FROM juholeinonen/kaldi-align:5.1

WORKDIR /home/app

# RUN apt-get update && apt-get dist-upgrade -y;
# RUN apt-get install -y python3;
# RUN pip3 install --upgrade pip
# RUN find /opt/kaldi -name "*.so*" | xargs strip; \
#    find /opt/intel ! -name "*.so*" -a -type f -delete;

RUN pip3 install Flask gunicorn==20.1.0 redis pydub requests
COPY . .
#COPY ./align_in_singularity.sh /opt/kaldi/egs/align/aligning_with_Docker/bin/align_in_singularity.sh
RUN chmod -R a+rwx /home/app; chmod -R a+rwx /opt/kaldi/egs;

ENTRYPOINT ./finnish-forced-align-init
#ENTRYPOINT []
