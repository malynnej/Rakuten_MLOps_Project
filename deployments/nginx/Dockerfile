FROM nginx:latest

COPY nginx.conf /etc/nginx/nginx.conf
COPY certs/ /etc/nginx/certs/
COPY .htpasswd /etc/nginx/.htpasswd

EXPOSE 80
EXPOSE 443
CMD ["nginx", "-g", "daemon off;"]
