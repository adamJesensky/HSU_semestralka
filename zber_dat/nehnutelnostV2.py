import time
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
import concurrent.futures
import logging
from tqdm import tqdm
# from fake_useragent import UserAgent
from torrequest import TorRequest
from stem import Signal
from stem.control import Controller
import os
from datetime import datetime, timedelta
from requests.exceptions import ProxyError, ConnectionError, Timeout
from sqlalchemy import JSON, create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
import random

from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
from stem.util import log

# Disable stem's logging below WARNING level
logging.getLogger('stem').setLevel(logging.WARNING)
log.get_logger().propagate = False  # Prevent stem from propagating logs


def generate_user_agent():
    # Definujeme typy prehliadačov a operačných systémov, ktoré chceme zahrnúť
    software_names      = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value, SoftwareName.EDGE.value, SoftwareName.SAFARI.value]
    operating_systems   = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value, OperatingSystem.MAC.value]
    
    # Inicializujeme generátor
    user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
    
    # Vyberieme náhodný user agent
    return user_agent_rotator.get_random_user_agent()

# Function to handle errors and log messages
def handle_error(message, exception=None):
    logging.error(message, exc_info=exception)
    print(message)

# Function to get a new Tor identity
def renew_tor_identity():
    try:
        with Controller.from_port(port=9051) as controller:
            controller.authenticate(password="fccfui24")  # Provide the correct authentication information if required
            controller.signal(Signal.NEWNYM)
            # Print the IP address after renewing identity
            #print("New IP Address:", requests.get("https://api64.ipify.org?format=json").json()["ip"])
    except Exception as e:
        pass  # Silently ignore Tor control errors

def get_current_date():
    return datetime.now().strftime("%Y%m%d")  # Corrected usage

def parse_date(date_str):
    """Convert a date string in the format 'YYYYMMDD' to a datetime.date object."""
    return datetime.strptime(date_str, "%Y%m%d").date()

def get_region_from_url(url):
    return url.split('/')[-2]


#added scrape_date, dates as datetime object
def scrape_propertyV5(url):
    try:
        for i in range(5):  # make up to 5 attempts with different User-Agent headers
            headers = {'User-Agent': generate_user_agent()}
            renew_tor_identity()  # Rotate Tor identity before each request
            property_response = requests.get(url, headers=headers, timeout=10)

            #property_response = requests.get(url, headers=headers)
            if property_response.status_code == 200:
                soup = BeautifulSoup(property_response.content, 'html.parser')
                # Extract information from JSON-LD scripts
                json_ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
                
                property_data = {'url': url, 'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                # Attempt to find and extract latitude and longitude from the map div
                map_div = soup.find('div', id='js-map-detail')
                if map_div:
                    property_data['latitude'] = map_div.get('data-latitude')
                    property_data['longitude'] = map_div.get('data-longitude')
                # Extract and store the main image URL
                og_image = soup.find('meta', property='og:image')
                if og_image and og_image.get('content'):
                    property_data['image_url'] = og_image['content']

                # Extract data-offer-cat-2(Rodinný dom) and data-offer-cat-4(Predaj) from the div with id 'page-info'
                page_info_div = soup.find('div', {'id': 'page-info'})
                if page_info_div:
                    property_data['druh_objektu_text'] = page_info_div.get('data-offer-cat-2', '')
                    property_data['typ'] = page_info_div.get('data-offer-cat-4', '')
                
                updated_div = soup.find('div', class_='updated')
                if updated_div:
                    spans = updated_div.find_all('span')
                    for span in spans:
                        text = span.text.strip()
                        if 'Aktualizovaný' in text:
                            updated_date_str = text.split(': ')[1].strip()
                            # Convert to datetime and then format as ISO 8601 string
                            updated_date = datetime.strptime(updated_date_str, '%d. %m. %Y').strftime('%Y-%m-%d')
                            property_data['updated_date'] = updated_date
                        elif '1. publikácia' in text:
                            publication_date_str = text.split(': ')[1].strip()
                            # Convert to datetime and then format as ISO 8601 string
                            publication_date = datetime.strptime(publication_date_str, '%d. %m. %Y').strftime('%Y-%m-%d')
                            property_data['publication_date'] = publication_date

                for script in json_ld_scripts:
                    try:
                        json_data = json.loads(script.string)
                        if isinstance(json_data, dict):
                            # Extract property information based on @type
                            if '@type' in json_data:
                                property_type = json_data.get('@type')
                                if property_type == 'BreadcrumbList':
                                    # Extract the word from position 2 DRUH OBJEKTU
                                    item_list_element = json_data.get('itemListElement', [])
                                    if len(item_list_element) >= 2:
                                        position_2_word = item_list_element[1].get('name', '')
                                        property_data['druh_objektu'] = position_2_word
                                
                                elif property_type == 'Product':
                                    property_data['name'] = json_data.get('name', '')
                                    property_data['price'] = json_data.get('offers', {}).get('price', '')
                                    property_data['description'] = json_data.get('description', '')

                                elif property_type == 'Residence':
                                     # Extract location data
                                    address = json_data.get('address', {})
                                    if address.get('@type') == 'PostalAddress':
                                        property_data['streetAddress'] = address.get('streetAddress', '')
                                        property_data['addressLocality'] = address.get('addressLocality', '')
                                        property_data['addressRegion'] = address.get('addressRegion', '')
                                   
                                elif property_type == 'SingleFamilyResidence':
                                    property_data['telephone'] = json_data.get('telephone', '')
                                    property_data['floor_size'] = json_data.get('floorSize', {}).get('value', '')
                                    # Extract amenityFeature
                                    amenity_feature = json_data.get('amenityFeature', [])
                                    for feature in amenity_feature:
                                        property_data[feature.get('name', '')] = feature.get('value', '')                       
                    except json.JSONDecodeError:
                        continue
                logging.info(f'Scraped : {url}')
                return property_data
            else:
                handle_error(f'Received status code {property_response.status_code} for {url}')
        return {}
    except Exception as e:
        handle_error(f'Error scraping property at {url}: {e}')
        return {}

# added checks if the page is valid, if not return empty list to signal end of valid pages
def scrape_page7(page_url, days_back=None, date_type='published'):
    max_retries = 9  # Maximum number of retries
    backoff_factor = 2  # Delay factor between retries
    retry_delay = 2  # Initial delay between retries in seconds

    for attempt in range(max_retries):
        try:
            headers = {'User-Agent': generate_user_agent()}
            renew_tor_identity()  # Rotate Tor identity before each request
            response = requests.get(page_url, headers=headers, timeout=10)
            
            # Check if we've been redirected to the main listings page
            if 'inzeraty' in response.url:
                return []  # Return empty list to signal end of valid pages
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Additional check - look for pagination element
            # pagination = soup.select_one('.pagination')
            # if not pagination:
            #     return []  # No pagination means we're not on a valid listings page
                
            offers = soup.select('.offer-item-in')
            if not offers:
                return []  # No offers found on page

            real_estate_urls = []
            for offer in offers:
                link = offer.select_one('.offer-body a')
                if not link or not link.get('href'):
                    continue
                    
                full_url = urljoin(page_url, link['href'])

                if days_back is not None:
                    offer_dates = offer.select('.offer-dates .offer-date')
                    within_date_range = False

                    for date in offer_dates:
                        date_text = date.get_text(strip=True)
                        try:
                            date_value = datetime.strptime(date_text.split(': ')[1], '%d.%m.%Y').date()
                            if (date_type == 'published' and 'Publikované:' in date_text) or (date_type == 'updated' and 'Aktualizované:' in date_text):
                                if (datetime.now().date() - date_value).days <= days_back:
                                    within_date_range = True
                                    break
                        except ValueError:
                            continue

                    if within_date_range:
                        real_estate_urls.append(full_url)
                else:
                    real_estate_urls.append(full_url)

            return real_estate_urls

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= backoff_factor
                continue
            else:
                handle_error(f'Error occurred while scraping page {page_url} after {max_retries} attempts: {e}', exception=True)
                return []
        except Exception as e:
            handle_error(f'An unexpected error occurred while scraping page {page_url}: {e}', exception=True)
            return []
         
def main4(scrape_mode='page', max_pages=10, days_back=None, date_type='published', save_mode='csv'):
    
    if scrape_mode == 'page':
        urls = ['https://www.reality.sk/zilinsky-kraj/?order=created_date-newest']
    elif scrape_mode == 'date':
        if date_type == 'published':
            urls = [
                'https://www.reality.sk/zilinsky-kraj/?order=created_date-newest',
                'https://www.reality.sk/trenciansky-kraj/?order=created_date-newest',
                'https://www.reality.sk/banskobystricky-kraj/?order=created_date-newest',
                'https://www.reality.sk/nitriansky-kraj/?order=created_date-newest',
                'https://www.reality.sk/bratislavsky-kraj/?order=created_date-newest',
                'https://www.reality.sk/trnavsky-kraj/?order=created_date-newest',
                'https://www.reality.sk/presovsky-kraj/?order=created_date-newest',
                'https://www.reality.sk/kosicky-kraj/?order=created_date-newest',
            ]
        elif date_type == 'updated':
            urls = [
                # 'https://www.reality.sk/zilinsky-kraj/?order=updated_date-newest',
                # 'https://www.reality.sk/trenciansky-kraj/?order=updated_date-newest',
                # 'https://www.reality.sk/banskobystricky-kraj/?order=updated_date-newest',
                # 'https://www.reality.sk/nitriansky-kraj/?order=updated_date-newest',
                'https://www.reality.sk/bratislavsky-kraj/?order=updated_date-newest',
                'https://www.reality.sk/trnavsky-kraj/?order=updated_date-newest',
                'https://www.reality.sk/presovsky-kraj/?order=updated_date-newest',
                'https://www.reality.sk/kosicky-kraj/?order=updated_date-newest'
            ]

    output_directory = 'nehnutelnosti_data'
    os.makedirs(output_directory, exist_ok=True)
    all_urls = []

    def process_page_batch(url, start_page, end_page):
        """Process a batch of pages in parallel"""
        region = get_region_from_url(url)
        batch_urls = []
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {
                executor.submit(scrape_page7, url + f'&page={page_num}', days_back, date_type): page_num 
                for page_num in range(start_page, end_page + 1)
            }
            
            for future in concurrent.futures.as_completed(futures):
                page_num = futures[future]
                try:
                    page_urls = future.result()
                    if page_urls:
                        batch_urls.extend(page_urls)
                        print(f"Processed page {page_num} in {region} - found {len(page_urls)} URLs")
                    else:
                        print(f"Empty page {page_num} in {region} - stopping condition")
                        return batch_urls, True  # Signal to stop
                except Exception as e:
                    handle_error(f'Error scraping page {page_num} in {region}: {e}')
        return batch_urls, False

    # Phase 1: Collect all URLs
    for url in urls:
        region = get_region_from_url(url)
        all_urls = []
        
        if scrape_mode == 'page':
            # Process all pages in parallel batches
            batch_size = 20
            for batch_start in range(1, max_pages + 1, batch_size):
                batch_end = min(batch_start + batch_size - 1, max_pages)
                batch_urls, should_stop = process_page_batch(url, batch_start, batch_end)
                all_urls.extend(batch_urls)
                if should_stop:
                    break

        elif scrape_mode == 'date':
            # For date mode, process in expanding batches until we find empty pages
            batch_size = 10
            current_page = 1
            while True:
                batch_urls, should_stop = process_page_batch(url, current_page, current_page + batch_size - 1)
                all_urls.extend(batch_urls)
                
                if should_stop or (days_back and len(batch_urls) < 15):  # Threshold for likely old results
                    break
                
                current_page += batch_size
                batch_size = min(batch_size * 2, 30)  # Gradually increase batch size up to 30

        # Phase 2: Process collected URLs for current region
        if all_urls:
            print(f"\nTotal URLs collected for {region}: {len(all_urls)}")
            print(f"Starting property data extraction for {region}...")
            
            with tqdm(total=len(all_urls), desc=f"Processing {region} properties") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                    futures = {executor.submit(scrape_propertyV5, url): url for url in all_urls}
                    all_nehnutelnosti_data = []
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            property_data = future.result()
                            if property_data:
                                all_nehnutelnosti_data.append(property_data)
                        except Exception as e:
                            handle_error(f'Error processing property: {e}')
                        finally:
                            pbar.update(1)

            # Save data for current region
            if save_mode == 'csv':
                filename = f'Nehnutelnosti_{region}_{get_current_date()}.csv'
                save_data_to_csv6(all_nehnutelnosti_data, filename, output_directory)
            elif save_mode == 'db':
                save_data_to_db(all_nehnutelnosti_data)

def save_data_to_csv6(data, filename, output_directory):
    try:
        df = pd.DataFrame(data)
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, filename)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"File exists. Updating {output_path}...")
            existing_data = pd.read_csv(output_path, low_memory=False)

            if existing_data.empty:
                print("Existing file is empty. Saving new data.")
                df.to_csv(output_path, index=False, header=True)
                return

            updated_count = 0
            added_count = 0

            for index, new_row in df.iterrows():
                new_row = new_row.reindex(existing_data.columns)
                url = new_row['url']
                updated_date_new = None

                if pd.notnull(new_row['updated_date']) and isinstance(new_row['updated_date'], str):
                    updated_date_new = datetime.strptime(new_row['updated_date'].strip(), '%d. %m. %Y').date()

                if url in existing_data['url'].values:
                    existing_row_index = existing_data[existing_data['url'] == url].index[0]
                    updated_date_existing_str = existing_data.at[existing_row_index, 'updated_date']
                    updated_date_existing = None

                    if pd.notnull(updated_date_existing_str) and isinstance(updated_date_existing_str, str):
                        updated_date_existing = datetime.strptime(updated_date_existing_str.strip(), '%d. %m. %Y').date()

                    if updated_date_new and (updated_date_existing is None or updated_date_new > updated_date_existing):
                        # Cast data types to match those of existing_data
                        for col in new_row.index:
                            if col in existing_data and pd.api.types.is_numeric_dtype(existing_data[col]):
                                new_row[col] = pd.to_numeric(new_row[col], errors='coerce')
                        existing_data.iloc[existing_row_index] = new_row
                        updated_count += 1
                else:
                    existing_data = pd.concat([existing_data, pd.DataFrame([new_row])], ignore_index=True)
                    added_count += 1

            existing_data.to_csv(output_path, index=False)
            print(f"Successfully updated {output_path}.")
            print(f"Records updated: {updated_count}, Records added: {added_count}")
        else:
            df.to_csv(output_path, index=False, header=True)
            print(f"Successfully saved {len(df)} entries to {output_path}")

    except Exception as e:
        print(f'Error saving data to {output_path}: {e}')

def combine_csv_files5(csv_files, output_file, input_directory):
    combined_data = pd.DataFrame()
    update_counts = 0
    add_counts = 0
    skipped_files = 0
    skipped_entries = 0
    processed_entries = 0
    duplicate_entries_within_files = 0

    for csv_file in csv_files:
        csv_path = os.path.join(input_directory, csv_file)
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            print(f"Processing file: {csv_file}")
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                total_entries = len(df)
                processed_entries += total_entries

                if df.empty:
                    print(f"Skipping file {csv_file} as it contains no data.")
                    skipped_files += 1
                    continue

                if 'url' not in df.columns or 'updated_date' not in df.columns:
                    print(f"Skipping file {csv_file} as it lacks 'url' or 'updated_date' columns.")
                    skipped_files += 1
                    skipped_entries += total_entries
                    continue

                df['updated_date'] = pd.to_datetime(df['updated_date'], errors='coerce', format='%d. %m. %Y')
                if df['updated_date'].isnull().all():
                    print(f"Skipping file {csv_file} as 'updated_date' column contains invalid data.")
                    skipped_files += 1
                    skipped_entries += total_entries
                    continue

                # Sort and remove duplicates within the file based on 'updated_date'
                initial_df_length = len(df)
                df.sort_values(by='updated_date', ascending=False, inplace=True)
                df.drop_duplicates(subset='url', keep='first', inplace=True)
                duplicate_entries_within_files += (initial_df_length - len(df))

                if combined_data.empty:
                    combined_data = df.copy()
                    add_counts += len(df)
                else:
                    df = df.reindex(columns=combined_data.columns)
                    url_set = set(combined_data['url'])

                    for index, row in df.iterrows():
                        url = row['url']
                        updated_date = row['updated_date']

                        if url in url_set:
                            existing_row_index = combined_data[combined_data['url'] == url].index[0]
                            existing_updated_date = combined_data.at[existing_row_index, 'updated_date']
                            
                            if pd.notnull(updated_date) and (pd.isnull(existing_updated_date) or updated_date > existing_updated_date):
                                # Align row to the columns of combined_data
                                row_aligned = row.reindex(combined_data.columns)
                                # Fill missing values with NaN or appropriate default values
                                row_aligned.fillna(value=np.nan, inplace=True)
                                # Ensure the DataFrame is large enough to accommodate the update
                                if existing_row_index < len(combined_data):
                                    combined_data.iloc[existing_row_index] = row_aligned
                                else:
                                    combined_data = pd.concat([combined_data, pd.DataFrame([row_aligned])], ignore_index=True)
                                update_counts += 1
                        else:
                            # Ensure that new row has the same columns as combined_data
                            row_aligned = row.reindex(combined_data.columns).fillna(value=np.nan)
                            combined_data = pd.concat([combined_data, pd.DataFrame([row_aligned])], ignore_index=True)
                            add_counts += 1
            except pd.errors.EmptyDataError:
                print(f"Skipping file {csv_file} as it appears to be empty or improperly formatted.")
                skipped_files += 1
                continue
            except Exception as e:
                print(f"Error processing file {csv_file}: {e}")
                skipped_files += 1
                continue
        else:
            print(f"Skipping file: {csv_file} as it is empty or does not exist.")
            skipped_files += 1

    if not combined_data.empty:
        # Convert 'updated_date' back to datetime for sorting
        combined_data['updated_date'] = pd.to_datetime(combined_data['updated_date'], errors='coerce', format='%d. %m. %Y')
        combined_data.sort_values(by='updated_date', ascending=False, inplace=True)

        initial_entries = len(combined_data)
        combined_data.drop_duplicates(subset='url', keep='first', inplace=True)
        unique_entries = len(combined_data)

        # Convert 'updated_date' back to the original format before saving
        combined_data['updated_date'] = combined_data['updated_date'].dt.strftime('%d. %m. %Y')

        os.makedirs(input_directory, exist_ok=True)
        output_path = os.path.join(input_directory, output_file)
        combined_data.to_csv(output_path, index=False)
        
        print(f"Combination Complete. {len(csv_files)} CSV files processed.")
        print(f"Initial entries: {initial_entries}")
        print(f"Updated entries: {update_counts}")
        print(f"Added new entries: {add_counts}")
        print(f"Skipped files: {skipped_files}")
        print(f"Skipped entries due to issues: {skipped_entries}")
        print(f"Processed entries: {processed_entries}")
        print(f"Duplicate entries removed within files: {duplicate_entries_within_files}")
        print(f"Duplicate entries removed in combined data: {initial_entries - unique_entries}")
        print(f"Unique entries after combination: {unique_entries}")
        print(f"Combined data saved to {output_path}")
    else:
        print("No valid CSV files to combine or all files lacked required columns.")

def save_data_to_db(data):
    session = Session()
    try:
        for item in data:
            new_listing = RealEstateListing(data = item)
            session.add(new_listing)
        session.commit()
    except Exception as e:
        print(f"Error saving data to DB: {e}")
        session.rollback()
    finally:
        session.close()



if __name__ == "__main__":
    output_directory = 'nehnutelnosti_data'
    os.makedirs(output_directory, exist_ok=True)
    
    # Database connection parameters
    db_config = {
        'database': 'Properties_01',
        'user': 'postgres',
        'password': 'fccfui24',
        'host': 'localhost',
        'port': 5432
    }

    # Create a SQLAlchemy engine
    db_connection = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    Session = sessionmaker(bind=db_connection)
    Base = declarative_base()
    
    class RealEstateListing(Base):
        __tablename__ = 'real_estate_listings'
        id = Column(Integer, primary_key=True)
        data = Column(JSONB)  
    Base.metadata.create_all(db_connection) 

    #main2(scrape_mode='date', days_back=1, date_type='updated')
    #main3(scrape_mode='page', max_pages=1, save_mode='db')
    #main3(scrape_mode='date', days_back=1, date_type='published',save_mode='db')
    # main3(scrape_mode='date', days_back=1, date_type='updated', save_mode='db')    
    
    
    # main3(scrape_mode='page',max_pages=300, save_mode='csv')    
    # main3(scrape_mode='date', days_back=365, date_type='updated', save_mode='csv')
    main4(scrape_mode='date', days_back=365, date_type='updated', save_mode='csv')
    #output_combined_file = f'combined_data_{get_current_date()}.csv'
    #combine_csv_files5(os.listdir(output_directory), output_combined_file, output_directory)





