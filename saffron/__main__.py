# saffron/__main__.py
import argparse
import saffron
import os
import platform
import requests
import tarfile
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


def GUI_init():
    print(
        "SAFFRON command line tool. It does nothing for now. In the future it will be linked to the GUI"
    )

def get_default_download_location():
    if platform.system() == "Windows":
        return os.path.join(os.getenv('USERPROFILE'), 'Downloads')
    else:
        return os.path.join(os.getenv('HOME'), 'Downloads')

def is_valid_download_link(url):
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('Content-Type')
        if response.status_code == 200 and 'application/x-gzip' in content_type:
            return True
        else:
            return False
    except Exception as e:
        print(f"{Fore.RED}Error checking the URL: {e}{Style.RESET_ALL}")
        return False

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    local_filename = os.path.join(dest_folder, url.split('/')[-1])
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(local_filename, 'wb') as f, tqdm(
        desc=local_filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            size = f.write(chunk)
            bar.update(size)
    
    return local_filename

def extract_tar_gz(file_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    total_size = os.path.getsize(file_path)
    with tarfile.open(file_path, "r:gz") as tar, tqdm(
        total=total_size,
        desc="Extracting",
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for member in tar.getmembers():
            tar.extract(member, path=extract_to)
            bar.update(member.size)

    os.remove(file_path)

def check_and_set_XUVTOP(extract_location):
    xuvtop = os.getenv('XUVTOP')
    if xuvtop:
        if os.path.abspath(xuvtop) == os.path.abspath(extract_location):
            print(f"{Fore.GREEN}The XUVTOP environment variable is already set to the correct location: {xuvtop}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}The XUVTOP environment variable is set to a different location: {xuvtop}{Style.RESET_ALL}")
            while True:
                choice = input(f"{Fore.YELLOW}Do you want to update it to the new location? (yes/no): {Style.RESET_ALL}").strip().lower()
                if choice == 'yes':
                    os.environ['XUVTOP'] = extract_location
                    update_XUVTOP_system_var(extract_location)
                    print(f"{Fore.GREEN}The XUVTOP environment variable has been updated to: {extract_location}{Style.RESET_ALL}")
                    break
                elif choice == 'no':
                    print(f"{Fore.YELLOW}The XUVTOP environment variable remains unchanged.{Style.RESET_ALL}")
                    break
                else:
                    print(f"{Fore.RED}Invalid input. Please enter 'yes' or 'no'.{Style.RESET_ALL}")
    else:
        os.environ['XUVTOP'] = extract_location
        update_XUVTOP_system_var(extract_location)
        print(f"{Fore.GREEN}The XUVTOP environment variable has been set to: {extract_location}{Style.RESET_ALL}")


def update_XUVTOP_system_var(extract_location):
    if platform.system() == "Windows":
        os.system(f'setx XUVTOP "{extract_location}"')
    else:
        bashrc_path = os.path.expanduser('~/.bashrc')

        # Read the current .bashrc file
        with open(bashrc_path, 'r') as file:
            lines = file.readlines()

        # Remove lines containing 'export XUVTOP'
        lines = [line for line in lines if 'export XUVTOP' not in line]

        # Add the new export line
        lines.append(f'\nexport XUVTOP="{extract_location}"\n')

        # Write the updated lines back to .bashrc
        with open(bashrc_path, 'w') as file:
            file.writelines(lines)

        # Source the ~/.bashrc file to apply changes immediately
        os.system(f'source ~/.bashrc')
        # os.system(f'export XUVTOP="{extract_location}"')


def setup_chianti_database():
    default_url = 'https://download.chiantidatabase.org/CHIANTI_10.1_database.tar.gz'
    default_location = get_default_download_location()

    print("\n----------------------------------------\n")

    print(f"Default download link is {Fore.GREEN}{default_url}")
    while True:
        url_choice = input(f"{Fore.YELLOW}Do you want to use the default download link? (yes/no): {Style.RESET_ALL}").strip().lower()
        
        if url_choice == 'yes':
            download_url = default_url
            break
        elif url_choice == 'no':
            custom_url = input(f"{Fore.YELLOW}Please specify the download link: {Style.RESET_ALL}").strip()
            if is_valid_download_link(custom_url):
                download_url = custom_url
                break
            else:
                print(f"{Fore.RED}The provided URL is not a valid download link or is not accessible. Please try again.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Invalid input. Please enter 'yes' or 'no'.{Style.RESET_ALL}")

    print("\n----------------------------------------\n")

    while True:
        print(f"Default download location is {Fore.GREEN}{default_location}")
        user_input = input(f"{Fore.YELLOW}Do you want to download the CHIANTI database file to this location? (yes/no/other): {Style.RESET_ALL}").strip().lower()
        
        if user_input == 'no':
            print(f"{Fore.RED}Download canceled by the user.{Style.RESET_ALL}")
            return
        elif user_input == 'other':
            custom_location = input(f"{Fore.YELLOW}Please specify the download location: {Style.RESET_ALL}").strip()
            if os.path.exists(custom_location):
                download_location = custom_location
                break
            else:
                print(f"{Fore.RED}The path '{custom_location}' does not exist. Please try again.{Style.RESET_ALL}")
        elif user_input == 'yes':
            download_location = default_location
            break
        else:
            print(f"{Fore.RED}Invalid input. Please enter 'yes', 'no', or 'other'.{Style.RESET_ALL}")

    print("\n----------------------------------------\n")

    tar_gz_file = os.path.join(download_location, download_url.split('/')[-1])
    extract_folder_name = os.path.splitext(os.path.splitext(tar_gz_file)[0])[0]
    extract_location = os.path.join(download_location, extract_folder_name)
    
    if os.path.exists(extract_location):
        print(f"The directory {Fore.GREEN}{extract_location}{Style.RESET_ALL} already exists.")
        print(f"{Fore.YELLOW}The CHIANTI database will not be downloaded again.")
        print(f"{Fore.YELLOW}If you want to re-download it, please remove the existing folder and rerun the script.")
        check_and_set_XUVTOP(extract_location)
        return

    if os.path.exists(tar_gz_file):
        print(f"{Fore.GREEN}The file {tar_gz_file} already exists.")
        print(f"{Fore.YELLOW}The file will not be downloaded again.")
    else:
        print(f"Downloading the file from {Fore.GREEN}{download_url}...")
        tar_gz_file = download_file(download_url, download_location)

    print("\n----------------------------------------\n")
    
    print(f"Extracting the file to {Fore.GREEN}{extract_location}{Style.RESET_ALL}...")
    extract_tar_gz(tar_gz_file, extract_location)

    print("\n----------------------------------------\n")
    
    print(f"{Fore.GREEN}File downloaded and extracted to {extract_location}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Download and extraction completed successfully.{Style.RESET_ALL}")

    check_and_set_XUVTOP(extract_location)

# if __name__ == '__main__':
#     main()

