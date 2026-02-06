#!/bin/bash

# This script generates .htpasswd files for the basic authentication at nginx.
# Only to be used for testing purposes, all users will have username=password.

TARGET_DIR="./deployments/nginx"


ADMINS_FN="admins.htpasswd"
ADMINS="admin1"

echo "Generating $ADMINS_FN ..."
touch ${TARGET_DIR}/${ADMINS_FN}
for user in $ADMINS
do
    htpasswd -b ${TARGET_DIR}/${ADMINS_FN} $user $user
done


USERS_FN="users.htpasswd"
USERS="user1 user2"

echo "Generating $USERS_FN ..."
touch ${TARGET_DIR}/${USERS_FN}
for user in $USERS
do
    htpasswd -b ${TARGET_DIR}/${USERS_FN} $user $user
done


DEVS_FN="devs.htpasswd"
DEVS="dev1 dev2"

echo "Generating $DEVS_FN ..."
touch ${TARGET_DIR}/${DEVS_FN}
for user in $DEVS
do
    htpasswd -b ${TARGET_DIR}/${DEVS_FN} $user $user
done
