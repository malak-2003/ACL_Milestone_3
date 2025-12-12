from neo4j import GraphDatabase

print("Script started.")

# Read config.txt
def load_config(path="data/config.txt"):
    config = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config[key.strip()] = value.strip()
    return config["URI"], config["USERNAME"], config["PASSWORD"]

# Test connection
def test_connection():
    uri, user, password = load_config()
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Connected!' AS message")
            print(result.single())
    except Exception as e:
        print("Connection FAILED:", e)
    finally:
        driver.close()

if __name__ == "__main__":
    test_connection()
