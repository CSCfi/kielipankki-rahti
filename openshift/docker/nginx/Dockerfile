FROM registry.access.redhat.com/ubi8/nginx-122

ADD nginx.conf "${NGINX_CONF_PATH}"
COPY start.sh /opt/app-root/start.sh

# start.sh writes the resolver configuration at runtime as an arbitrary user.
USER 0
RUN chmod +x /opt/app-root/start.sh && chmod g+rwX "${NGINX_CONFIGURATION_PATH}"
USER 1001

CMD ["/opt/app-root/start.sh"]
