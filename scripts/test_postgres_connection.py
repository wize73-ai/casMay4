#!/usr/bin/env python3
"""
Test PostgreSQL Connection with Various Credentials
"""

import sys
import psycopg2
import time

# Define the host and port
host = "192.168.1.105"
port = 5432

# Define potential usernames and passwords to try
usernames = ["postgres", "pi", "admin", "casalingua"]
passwords = ["postgres", "raspberry", "admin", "password", "casalingua"]

# Flag to indicate if any connection succeeded
connection_succeeded = False

print(f"Testing PostgreSQL connection to {host}:{port}...")
print("=================================================")

for username in usernames:
    for password in passwords:
        try:
            conn_string = f"host={host} port={port} user={username} password={password} dbname=postgres connect_timeout=5"
            print(f"Trying connection as user '{username}' with password '{password}'...")
            
            # Attempt to connect
            conn = psycopg2.connect(conn_string)
            
            # If we get here, connection succeeded
            print(f"\n✅ CONNECTION SUCCESSFUL!")
            print(f"Successfully connected to PostgreSQL server as user '{username}'")
            
            # Get server version
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"PostgreSQL Server Version: {version}")
            
            # List databases
            cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
            databases = cursor.fetchall()
            print(f"\nAvailable databases:")
            for db in databases:
                print(f"  - {db[0]}")
                
            # Close the connection
            cursor.close()
            conn.close()
            
            # Construct the successful connection string for CasaLingua
            success_string = f"postgresql://{username}:{password}@{host}:{port}/casalingua"
            print(f"\nConnection string for CasaLingua:")
            print(f"{success_string}")
            
            connection_succeeded = True
            break
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
        
        # Small delay between attempts
        time.sleep(0.5)
    
    if connection_succeeded:
        break

if not connection_succeeded:
    print("\n❌ All connection attempts failed.")
    print("Please verify the PostgreSQL server is running and credentials are correct.")
    sys.exit(1)
else:
    print("\nTest completed successfully.")
    sys.exit(0)