# ------------------------------------------------------------------------------
# Check connection by sending messages to a Telegram Bot once in a while.
# ------------------------------------------------------------------------------

from mp.utils.update_bots.telegram_bot import TelegramBot
from mp.paths import telegram_login
import time

def run_for_mins(bot, nr_mins):
    r"""Run for an many minutes, giving updates once per minute."""
    for i in range(1, nr_mins+1):
        time.sleep(60)
        bot.send_msg('It has been {} minutes.'.format(i))

def run_for_hours(bot, nr_hours):
    r"""Run for an many hours, giving updates once per hour."""
    for i in range(1, nr_hours+1):
        time.sleep(360)
        bot.send_msg('It has been {} hours'.format(i))

"""
bot = TelegramBot(login_data=telegram_login)
nr_hs = 14
bot.send_msg('Connection testing script started. Should run for {} hours.'.format(nr_hs))
run_for_hours(bot, nr_hs)
"""