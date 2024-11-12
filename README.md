# Generate Auto Assign Certificate using OpenSSL

1. Install OpenSSL

2. Run

    ```bash    
    $-> openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout certificates/server.key -out certificates/server.crt
    ```
