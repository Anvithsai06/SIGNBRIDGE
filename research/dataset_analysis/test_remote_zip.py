import json
import ssl
import certifi
from urllib.request import Request, urlopen

API_URL = "https://zenodo.org/api/records/4010759"

context = ssl.create_default_context(
    cafile=certifi.where()
)


def main():

    print("Connecting to Zenodo...")

    with urlopen(
        API_URL,
        context=context,
        timeout=60
    ) as response:

        data = json.loads(response.read().decode())

    target = next(
        file for file in data["files"]
        if file["key"] == "Seasons_1of1.zip"
    )

    url = target["links"]["self"]

    print("\nTesting archive:", target["key"])
    print("Archive size:", target["size"])

    request = Request(
        url,
        headers={
            "Range": "bytes=-65536"
        }
    )

    with urlopen(
        request,
        context=context,
        timeout=60
    ) as response:

        print("\n========== HTTP RESPONSE ==========")

        print("Status:", response.status)
        print("Content-Length:", response.headers.get("Content-Length"))
        print("Content-Range:", response.headers.get("Content-Range"))
        print("Accept-Ranges:", response.headers.get("Accept-Ranges"))

        data = response.read()

        print("Bytes received:", len(data))

        if response.status == 206:
            print("\nSUCCESS: Server supports partial content.")
        else:
            print("\nPartial content was not confirmed.")


if __name__ == "__main__":
    main()