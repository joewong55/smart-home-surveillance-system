# Owner: Joseph Wong
# Last updated: 3/21/20
import globals as g

from twilio.rest import Client

def send_alert():
    """
    Description: Sends SMS alert using Twilio
    Parameters: None
    Return: None
    """

    while(True):
        if g.facial_alert_queue and g.package_alert_queue: # loop through global lists in order if they ar enot empty
            if g.facial_alert_queue[0]==g.package_alert_queue[0]:
                # Handle cases
                if g.facial_alert_queue[1]=='no_person':
                    msg = 'No one detected'
                    alert = 0
                elif g.facial_alert_queue[1]=='unknown_person' and g.package_alert_queue[1]=='package':
                    msg = 'Detected unknown person with a package!'
                    alert = 1
                elif g.facial_alert_queue[1]=='unknown_person' and g.package_alert_queue[1]=='no_package':
                    msg = 'Detected unknown person!'
                    alert = 1
                elif g.facial_alert_queue[1]=='known_person' and g.package_alert_queue[1]=='package':
                    msg = 'Detected known person with a package'
                    alert = 0
                elif g.facial_alert_queue[1]=='known_person' and g.package_alert_queue[1]=='no_package':
                    msg = 'Detected known person with no package'
                    alert = 0

                print(msg + " from frame: " + str(g.facial_alert_queue[0]))

                del g.facial_alert_queue[:2] # remove from global list
                del g.package_alert_queue[:2]

                if alert:
                    # use credentials from Twilio account
                    client = Client("<Twilio Account SID>", "<Auth Token>")

                    # send SMS alert
                    send_to = "1234567890"
                    sent_from = "0987654321"
                    client.messages.create(to="+1" + send_to,
                                           from_="+1" + sent_from,
                                           body=msg)
