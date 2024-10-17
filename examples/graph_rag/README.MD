# Run Neo4j in a Docker container

    docker run --publish=7474:7474 --publish=7687:7687 --volume=$HOME/neo4j/data:/data -e NEO4J_AUTH=none neo4j

# Download the BBC news dataset

    wget https://raw.githubusercontent.com/amankharwal/Website-data/master/bbc-news-data.csv