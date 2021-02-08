import requests
import main_core

token = '1455917480:AAGBMH9gLyzlH_zn811OBqLnDdtXcSC29I8'

api_url = "https://api.telegram.org/bot" + token
updates_url = api_url + "/getUpdates"


def get_url(temp):
    offset = temp
    url = updates_url + "?offset=" + str(offset) + "&timeout=100"
    return url


def send_msg(msg, ch_id):
    return requests.post(api_url + "/SendMessage" + "?chat_id=" + str(
        ch_id) + "&text=" + msg)


def last_update():
    resp = requests.get(updates_url)
    out = resp.json()['result']
    lst_updt = out[len(resp.json()['result']) - 1]
    return lst_updt['update_id'] + 1


ofst = last_update()


while True:
    response = requests.get(get_url(ofst))
    if len(response.json()['result']) > 0:
        msg = response.json()['result'][0]
        var = msg['message']['text']

        val = main_core.get_emotion(var)
        ret_msg = f"belonging emotion category is '{val}'"
        chat_id = msg['message']['chat']['id']
        send_msg(ret_msg, chat_id)
        ofst += 1

