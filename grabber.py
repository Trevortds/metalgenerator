
import requests
import re
import os
import subprocess
import csv
import time






albums = ['sabaton/fistforfight', 'sabaton/primovictoria', 'sabaton/atterodominatus', 'sabaton/metalizer', 'sabaton/theartofwar', 'sabaton/coatofarms', 'sabaton/carolusrex', 'sabaton/heroes', 'sabaton/thelaststand',
		  'dreamtheater/whendreamanddayunite', 'dreamtheater/imagesandwords', 'dreamtheater/awake', 'dreamtheater/achangeofseasons', 'dreamtheater/fallingintoinfinity', 'dreamtheater/scenesfromamemory', 'dreamtheater/sixdegreesofinnerturbulence', 'dreamtheater/trainofthought', 'dreamtheater/octavarium', 'dreamtheater/score', 'dreamtheater/systematicchaos', 'dreamtheater/blackcloudssilverlinings', 'dreamtheater/adramaticturnofevents', 'dreamtheater/dreamtheater', 'dreamtheater/illuminationtheory', 'dreamtheater/theastonishing',
		  'blindguardian/lucifersheritagesymphoniesofdoom', 'blindguardian/battalionsoffear', 'blindguardian/followtheblind', 'blindguardian/talesfromthetwilightworld', 'blindguardian/somewherefarbeyond', 'blindguardian/imaginationsfromtheotherside', 'blindguardian/nightfallinmiddleearth', 'blindguardian/donttalktostrangers', 'blindguardian/anightattheopera', 'blindguardian/atwistinthemyth', 'blindguardian/fly', 'blindguardian/anotherstrangerme', 'blindguardian/attheedgeoftime', 'blindguardian/beyondtheredmirror',
		  'ayreon/thefinalexperiment', 'ayreon/actualfantasy', 'ayreon/intotheelectriccastle', 'ayreon/ayreonautsonly', 'ayreon/theuniversalmigratorpartithedreamsequencer', 'ayreon/theuniversalmigratorpartiiflightofthemigrator', 'ayreon/loser', 'ayreon/thehumanequation', 'ayreon/01011001', 'ayreon/thetheoryofeverything', 'ayreon/thesource', 'ayreon/nonalbumsongs',
		  'alestorm/captainmorgansrevenge', 'alestorm/leviathan', 'alestorm/blacksailsatmidnight', 'alestorm/backthroughtime', 'alestorm/inthenavy', 'alestorm/sunsetonthegoldenage', 'alestorm/nogravebutthesea',
		  'haken/aquarius', 'haken/visions', 'haken/themountain', 'haken/restoration', 'haken/affinity',
		  'kamelot/eternity', 'kamelot/dominion', 'kamelot/sigeperilous', 'kamelot/thefourthlegacy', 'kamelot/karma', 'kamelot/epica', 'kamelot/theblackhalo', 'kamelot/onecoldwintersnight', 'kamelot/ghostopera', 'kamelot/poetryforthepoisoned', 'kamelot/silverthorn', 'kamelot/haven',
		  'gravedigger/heavymetalbreakdown', 'gravedigger/shootherdown', 'gravedigger/metalattackvol1helloweencelticfrostrunningwildgravediggersinnerwarrant', 'gravedigger/witchhunter', 'gravedigger/wargames', 'gravedigger/strongerthanever', 'gravedigger/thereaper', 'gravedigger/symphonyofdeath', 'gravedigger/heartofdarkness', 'gravedigger/tunesofwar', 'gravedigger/knightsofthecross', 'gravedigger/excalibur', 'gravedigger/thegravedigger', 'gravedigger/rheingold', 'gravedigger/thelastsupper', 'gravedigger/libertyordeath', 'gravedigger/pray', 'gravedigger/balladsofahangman', 'gravedigger/theclanswillriseagain', 'gravedigger/homeatlast', 'gravedigger/clashofthegods', 'gravedigger/returnofthereaper', 'gravedigger/healedbymetal', 'gravedigger/nonalbumsongs',
		  'seventhwonder/seventhwonder', 'seventhwonder/become', 'seventhwonder/waitinginthewings', 'seventhwonder/mercyfalls', 'seventhwonder/thegreatescape', 'seventhwonder/innerenemy', 'seventhwonder/welcometoatlantalive2014',
		  'asoundofthunder/outofthedarkness', 'asoundofthunder/queenofhell', 'asoundofthunder/timesarrow', 'asoundofthunder/thelesserkeyofsolomon', 'asoundofthunder/talesfromthedeadside',
		  'vancanto/astormtocome', 'vancanto/hero', 'vancanto/tribeofforce', 'vancanto/breakthesilence', 'vancanto/dawnofthebrave', 'vancanto/voicesoffire',
		  'symphonyx/symphonyx', 'symphonyx/thedamnationgame', 'symphonyx/thedivinewingsoftragedy', 'symphonyx/preludetothemillennium', 'symphonyx/twilightinolympus', 'symphonyx/vthenewmythologysuite', 'symphonyx/liveontheedgeofforever', 'symphonyx/theodyssey', 'symphonyx/paradiselost', 'symphonyx/iconoclast', 'symphonyx/underworld',
		  'shadowgallery/shadowgallery', 'shadowgallery/carvedinstone', 'shadowgallery/tyranny', 'shadowgallery/legacy', 'shadowgallery/roomv', 'shadowgallery/digitalghosts',
		  'devintownsend/oceanmachine', 'devintownsend/christeen', 'devintownsend/infinity', 'devintownsend/asssordiddemos119901996', 'devintownsend/physicist', 'devintownsend/terria', 'devintownsend/acceleratedevolution', 'devintownsend/synchestra', 'devintownsend/ziltoidtheomniscient', 'devintownsend/addicted', 'devintownsend/ki', 'devintownsend/deconstruction', 'devintownsend/ghost', 'devintownsend/epicloud', 'devintownsend/z2', 'devintownsend/transcendence',
		  'ironmaiden/ironmaiden', 'ironmaiden/runningfree', 'ironmaiden/womeninuniform', 'ironmaiden/killers', 'ironmaiden/thenumberofthebeast', 'ironmaiden/pieceofmind', 'ironmaiden/powerslave', 'ironmaiden/somewhereintime', 'ironmaiden/wastedyears', 'ironmaiden/caniplaywithmadness', 'ironmaiden/seventhsonofaseventhson', 'ironmaiden/noprayerforthedying', 'ironmaiden/bequickorbedead', 'ironmaiden/fearofthedark', 'ironmaiden/fromhereforeternity', 'ironmaiden/manontheedge', 'ironmaiden/thexfactor', 'ironmaiden/virus', 'ironmaiden/virtualxi', 'ironmaiden/bravenewworld', 'ironmaiden/danceofdeath', 'ironmaiden/nomoreliesdanceofdeath', 'ironmaiden/rainmaker', 'ironmaiden/wildestdreams', 'ironmaiden/amatteroflifeanddeath', 'ironmaiden/thefinalfrontier', 'ironmaiden/thebookofsouls',
]


albums = [
'seventhwonder/welcometoatlantalive2014', 
'asoundofthunder/outofthedarkness', 
'asoundofthunder/queenofhell', 
'asoundofthunder/timesarrow', 
'asoundofthunder/thelesserkeyofsolomon', 
'asoundofthunder/talesfromthedeadside', 
'vancanto/astormtocome', 
'vancanto/hero', 
'vancanto/tribeofforce', 
'vancanto/breakthesilence', 
'vancanto/dawnofthebrave', 
'vancanto/voicesoffire', 
'symphonyx/symphonyx', 
'symphonyx/thedamnationgame', 
'symphonyx/thedivinewingsoftragedy', 
'symphonyx/preludetothemillennium', 
'symphonyx/twilightinolympus', 
'symphonyx/vthenewmythologysuite', 
'symphonyx/liveontheedgeofforever', 
'symphonyx/theodyssey', 
'symphonyx/paradiselost', 
'symphonyx/iconoclast', 
'symphonyx/underworld', 
'shadowgallery/shadowgallery', 
'shadowgallery/carvedinstone', 
'shadowgallery/tyranny', 
'shadowgallery/legacy', 
'shadowgallery/roomv', 
'shadowgallery/digitalghosts', 
'devintownsend/oceanmachine', 
'devintownsend/christeen', 
'devintownsend/infinity', 
'devintownsend/asssordiddemos119901996', 
'devintownsend/physicist', 
'devintownsend/terria', 
'devintownsend/acceleratedevolution', 
'devintownsend/synchestra', 
'devintownsend/ziltoidtheomniscient', 
'devintownsend/addicted', 
'devintownsend/ki', 
'devintownsend/deconstruction', 
'devintownsend/ghost', 
'devintownsend/epicloud', 
'devintownsend/z2', 
'devintownsend/transcendence', 
'ironmaiden/ironmaiden', 
'ironmaiden/runningfree', 
'ironmaiden/womeninuniform', 
'ironmaiden/killers', 
'ironmaiden/thenumberofthebeast', 
'ironmaiden/pieceofmind', 
'ironmaiden/powerslave', 
'ironmaiden/somewhereintime', 
'ironmaiden/wastedyears', 
'ironmaiden/caniplaywithmadness', 
'ironmaiden/seventhsonofaseventhson', 
'ironmaiden/noprayerforthedying', 
'ironmaiden/bequickorbedead', 
'ironmaiden/fearofthedark', 
'ironmaiden/fromhereforeternity', 
'ironmaiden/manontheedge', 
'ironmaiden/thexfactor', 
'ironmaiden/virus', 
'ironmaiden/virtualxi', 
'ironmaiden/bravenewworld', 
'ironmaiden/danceofdeath', 
'ironmaiden/nomoreliesdanceofdeath', 
'ironmaiden/rainmaker', 
'ironmaiden/wildestdreams', 
'ironmaiden/amatteroflifeanddeath', 
'ironmaiden/thefinalfrontier', 
'ironmaiden/thebookofsouls', 

]



# use this from the interpreter/command line to get album titles
# album_regex = re.compile("/lyrics/(.*).html#1\"")
# r = requests.get("http://www.darklyrics.com/s/sabaton.html") # change this line appropriately
# print(re.findall(album_regex, r.text))




lyrics_regex = re.compile("class=\"lyrics\">((?:\n|.)*)<div class=\"(?:note|thank)")
html_regex = re.compile("<.*>")

with open("lyrics.txt", 'a') as writefile:
	for album in albums:
		time.sleep(3.2692)
		r = requests.get("http://www.darklyrics.com/lyrics/" + album + ".html")
		webpage = r.text;

		lyrics = re.findall(lyrics_regex, webpage)
		if len(lyrics) < 1:
			print("the following album didn't match the regex")
			print(album)
			continue

		lyrics = re.sub(html_regex, "", lyrics[0])

		writefile.write(lyrics)









# alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

# alphabet = ["a"] #for debugging

# num_pages = {'a':59, 'b':52, 'c':83, 'd':45, 'e':37, 'f':41, 'g':34, 'h':36, 'i':36, 'j':9, 'k':8, 
# 	'm':55, 'n':19, 'o':21, 'p':80, 'q':5, 'r':45, 's':107, 't':45, 'u':23, 'v':14, 'w':21, 'x':0, 'y':3, 'z':2}

# num_pages = {'a':5} #for debugging










# word_regex = re.compile('allowed_in_frame=0\">([a-zA-Z\(\)0-9 / -\.\&\;]*)</a> <a href=') # grabs the title of each entry from html
# word_regex = re.compile('allowed_in_frame=0\">(.*)</a> <a href=') # grabs the title of each entry from html
# etym_regex = re.compile('(<dd.*?>(?:.|\n)*?</dd>)')
# html_purge_regex = re.compile('>(.*?)<')
# from_regex = re.compile('from (([A-Z][a-zA-Z-]*) )+')

# with open('etymonline.tsv', 'w') as csvfile:
# 	etymwriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
# 	for letter in alphabet:
# 		for i in range(0, num_pages[letter]):
# 			r = requests.get("http://www.etymonline.com/index.php?l=" + letter + "&p=" + str(i) + "&allowed_in_frame=0")
# 			webpage = r.text;
# 			list_of_words = re.findall(word_regex, webpage)
# 			list_of_etyms = re.findall(etym_regex, webpage)
# 			#print(list_of_etyms)
# 			print(list_of_words)
# 			if len(list_of_etyms) != len(list_of_words):
# 				print(list_of_words)
# 				raise Exception("definition/wordlist mismatch at "+ letter+ str(i)+
# 					": \n "+ str(len(list_of_words))+ " words "+ str(len(list_of_etyms))+" definitions")

# 			for j in range(0, len(list_of_etyms)):

# 				etymwriter.writerow([list_of_words[j], list_of_etyms[j]])