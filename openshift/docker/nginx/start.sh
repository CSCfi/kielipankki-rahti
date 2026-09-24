#!/bin/sh
# Point nginx at the cluster DNS server so that upstream Service names are
# resolved per request rather than once at startup. nginx's resolver ignores
# the search domains in /etc/resolv.conf, so the upstreams are written as
# fully qualified names with this pod's namespace.
set -e
nameserver=$(awk '/^nameserver/ { print $2; exit }' /etc/resolv.conf)
namespace=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null \
    || awk '/^search/ { sub(/\.svc\..*/, "", $2); print $2; exit }' /etc/resolv.conf)
cat > /opt/app-root/etc/nginx.d/resolver.conf <<CONF
resolver ${nameserver:-127.0.0.1} valid=30s ipv6=off;
map \$host \$svc_domain {
    default "${namespace:-default}.svc.cluster.local";
}
CONF
exec nginx -g "daemon off;"
