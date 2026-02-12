FROM nginx:1.29.4

COPY nginx.conf /etc/nginx/nginx.conf
COPY certs/ /etc/nginx/certs/
COPY *.htpasswd /etc/nginx/

EXPOSE 80
EXPOSE 443
CMD ["nginx", "-g", "daemon off;"]
