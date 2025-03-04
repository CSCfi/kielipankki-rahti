FROM registry.access.redhat.com/ubi8/nginx-122

ADD nginx.conf "${NGINX_CONF_PATH}"
#ADD site.conf "${NGINX_DEFAULT_CONF_PATH}"
#ADD upstream.conf "${NGINX_CONFIGURATION_PATH}"

# Run script uses standard ways to run the application
CMD nginx -g "daemon off;"
