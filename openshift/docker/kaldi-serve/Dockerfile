FROM image-registry.openshift-image-registry.svc:5000/kielipankki-services/kaldi-serve:latest-py3.6

#FROM vernacularai/kaldi-serve:latest-py3.6
#FROM kaldi-serve:py3.6

WORKDIR /home/app

#RUN useradd -rm -d /home/gunicorn -s /bin/bash -g root -G sudo -u 1001 gunicorn
#RUN mv /root/.pyenv /home/gunicorn/.pyenv; for filename in $(find /home/gunicorn/.pyenv -type f); do sed -i 's/\/root\//\/home\/gunicorn\//' $filename; done

ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
#ENV PYENV_ROOT /home/gunicorn/.pyenv
#ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
#RUN chown -R gunicorn $PYENV_ROOT; chmod -R a+rx $PYENV_ROOT;
RUN chmod -R a+rx /root;

RUN apt-get -y install vim curl;

# RUN find /opt/kaldi -name "*.so*" | xargs strip; \
#     find /opt/kaldi ! -name "*.so*" -a -type f -delete; \
#     find /opt/intel ! -name "*.so*" -a -type f -delete; \
#     apt-get --purge -y remove cmake make automake g++ libc++-dev libboost-all-dev libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git binfmt-support bzip2-doc cmake-data fakeroot g++-6 git-man icu-devtools libalgorithm-diff-perl libalgorithm-diff-xs-perl libalgorithm-merge-perl libarchive13 libboost-atomic-dev libboost-atomic1.62-dev; \
#     apt-get --purge -y remove libboost-atomic1.62.0 libboost-chrono-dev libboost-chrono1.62-dev \
#   libboost-chrono1.62.0 libboost-context-dev libboost-context1.62-dev \
#   libboost-context1.62.0 libboost-coroutine-dev libboost-coroutine1.62-dev \
#   libboost-coroutine1.62.0 libboost-date-time-dev libboost-date-time1.62-dev \
#   libboost-date-time1.62.0 libboost-dev libboost-exception-dev \
#   libboost-exception1.62-dev libboost-fiber-dev libboost-fiber1.62-dev \
#   libboost-fiber1.62.0 \
#   libboost-graph-dev libboost-graph-parallel-dev \
#   libboost-graph-parallel1.62-dev libboost-graph-parallel1.62.0 \
#   libboost-graph1.62-dev libboost-graph1.62.0 libboost-iostreams1.62.0 \
#   libboost-locale-dev libboost-locale1.62-dev libboost-locale1.62.0 \
#   libboost-log1.62.0 libboost-math-dev libboost-math1.62-dev \
#   libboost-math1.62.0 libboost-mpi-dev libboost-mpi-python-dev \
#   libboost-mpi-python1.62-dev libboost-mpi-python1.62.0 libboost-mpi1.62-dev \
#   libboost-mpi1.62.0 libboost-program-options-dev \
#   libboost-program-options1.62-dev libboost-program-options1.62.0 \
#   libboost-python-dev libboost-python1.62-dev libboost-python1.62.0 \
#   libboost-random-dev libboost-random1.62-dev libboost-random1.62.0 \
#   libboost-regex1.62.0 libboost-serialization-dev \
#   libboost-serialization1.62-dev libboost-serialization1.62.0 \
#   libboost-signals-dev libboost-signals1.62-dev libboost-signals1.62.0 \
#   libboost-system-dev libboost-system1.62-dev \
#   libboost-test-dev libboost-test1.62-dev libboost-test1.62.0 \
#   libboost-thread1.62.0 libboost-timer-dev libboost-timer1.62-dev \
#   libboost-timer1.62.0 libboost-tools-dev libboost-type-erasure-dev \
#   libboost-type-erasure1.62-dev libboost-type-erasure1.62.0 libboost-wave-dev \
#   libboost-wave1.62-dev libboost-wave1.62.0 libboost1.62-dev \
#   libboost1.62-tools-dev libc++-helpers libc++1 libcurl3 libdpkg-perl libedit2 \
#   liberror-perl libexpat1-dev libfabric1 libfakeroot libfile-fcntllock-perl \
#   libfontenc1 libhwloc-dev libhwloc-plugins libhwloc5 libibverbs-dev \
#   libibverbs1 libice-dev libjsoncpp1 libllvm3.8 liblocale-gettext-perl \
#   liblzo2-2 libncurses5 libnl-3-200 libnl-route-3-200 libnuma-dev \
#   libopenmpi-dev libopenmpi2 libpciaccess0 libpipeline1 libpng-tools \
#   libprocps6 libpsm-infinipath1 libpthread-stubs0-dev libpython-dev \
#   libpython-stdlib libpython2.7 libpython2.7-dev libpython3-dev libpython3.5 \
#   libpython3.5-dev librdmacm1 libsm-dev libssl-doc libssl1.0.2 libstdc++-6-dev \
#   libtcl8.6 libtinfo-dev libtk8.6 libutempter0 libuv1 libx11-dev libx11-doc \
#   libxau-dev libxaw7 libxcb1-dev libxcomposite1 libxdmcp-dev libxext-dev \
#   libxft2 libxmu6 libxmuu1 libxpm4 libxrender-dev libxss-dev libxt-dev libxt6 \
#   libxxf86dga1 mpi-default-bin mpi-default-dev ocl-icd-libopencl1 openmpi-bin \
#   openmpi-common pkg-config procps psmisc python python-cffi-backend \
#   python-cryptography python-dev python-enum34 python-idna python-ipaddress \
#   python-minimal python-pkg-resources python-pyasn1 python-setuptools \
#   python-six python2.7-dev python3-dev python3.5-dev tcl tcl8.6 tk tk8.6 \
#   x11-utils x11proto-core-dev x11proto-input-dev x11proto-kb-dev \
#   x11proto-render-dev x11proto-scrnsaver-dev x11proto-xext-dev xbitmaps \
#   xorg-sgml-doctools xterm xtrans-dev libboost-filesystem-dev libboost-filesystem1.62-dev ;

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
