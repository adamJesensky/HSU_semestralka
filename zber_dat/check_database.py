from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import JSON, create_engine, Column, Integer, String, Float, DateTime


# Database connection setup
db_config = {
    'database': 'Properties_01',
    'user': 'postgres',
    'password': 'fccfui24',  # Change to your actual password
    'host': 'localhost',
    'port': 5432
}

engine = create_engine(f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
Session = sessionmaker(bind=engine)
Base = declarative_base()

class RealEstateListing(Base):
    __tablename__ = 'real_estate_listings'
    id = Column(Integer, primary_key=True)
    data = Column(JSONB)

# Create a session
session = Session()

# Query the database
try:
    listings = session.query(RealEstateListing).limit(10).all()
    for listing in listings:
        print(listing.data)
finally:
    session.close()