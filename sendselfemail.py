import smtplib
import sys

def sendmsg(msg = "Job's done!"): 
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login("trevortds3@gmail.com", "CfA^5dofO801u6^k")
	 

	server.sendmail("trevortds3@gmail.com", "trevortds@gmail.com", msg)
	server.quit()

if __name__ == '__main__':
	sendmsg(sys.argv[1])