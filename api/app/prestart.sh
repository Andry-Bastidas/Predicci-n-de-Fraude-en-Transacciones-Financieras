

echo " Waitinf for postgres connection"

while ! nc -z db 5432; do
    sleep 0.1

done

echo "PostgresSQL started

exec "$@"