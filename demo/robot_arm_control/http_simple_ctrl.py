import requests
import argparse


def main():
    parser = argparse.ArgumentParser(description='Http JSON Communication')
    parser.add_argument('ip', type=str, help='IP address: 192.168.10.104')
    args = parser.parse_args()
    ip_addr = args.ip

    try:
        while True:
            command = input("input your json cmd: ")
            url = "http://" + ip_addr + "/js?json=" + command
            response = requests.get(url)
            content = response.text
            print(content)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
