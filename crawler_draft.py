import requests

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Image successfully downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading image: {e}")

if __name__ == "__main__":
    image_url = "https://cms.iproyal.com/uploads/header_image_e97e0a28ed.webp"  # Replace with your image URL
    download_image(image_url, "downloaded_image.webp")