# -----------
# DESCRIPTION
# -----------
In general, the data should be in the self-explaining NetCDF format again, just
like the instrumental data. There are 30 ensemble member, i.e. 30 simulations
run from the year 1600 to 2005 C.E. The ensemble member can be seen in the file
name. There is one file per year with 12 monthly means/sums of multiple
variables in each file, e.g.: temperature at 2 m above ground and at the 500
hPa pressure level as well as precipitations amount, sea level pressure, etc.

# --------
# DOWNLOAD
# --------
1) We start by getting all the links to the files (techniques from web
                                                   scraping, its fun, there
                                                   used to be competitions to
                                                   scrap Amazon back in the
                                                   days 😊 (competitors then
                                                              use the scraped
                                                              data to adapt
                                                              thei pricing one
                                                              notch below))

  curl http://giub-torrent.unibe.ch/DATA/REUSE/CCC400_ensmean/ | grep -i nc |
sed -n 's/.*href="\([^"]*\).*/\1/p' >> urls_ensmean.txt

This basically scraps all the links (the hrefs) from the page, cleans the
strings and puts them in a file called urls_ensmean.txt.

2) Now you can use this file to generate the urls pointing to each file and
download them.

  cat urls_ensmean.txt | parallel -j 1 --gnu "wget
http://giub-torrent.unibe.ch/DATA/REUSE/CCC400_ensmean/{}"

This command just prints the lines in the file (cat) then dispatches each url
to wget (web-get, download tool) using GNUParallel (a really powerful tool,
                                                    worth checking).
You can change the amount of parallelization using the option in purple.
I.e. you can use -j 20 to run 20 downloads in parallel (this should be fine).

--------------------------------------------------------------------------------------------------

Note that this will only download the ensemble means CCC400_ensmean. There is
another folder containing the ensemble members.
You can download it by re-using the above proceduce, but replacing all
instances of CCC400_ensmean with CCC400_ens_mem.
Also change the name of the file used to store the urls to urls_ens_mem.txt so
we keep everything.

FINAL NOTE: Since the full dataset is quite heavy, you can download only parts
of it by editing the ursl file after step 1) to only keep a subset of the data
(just select and delete the lines that you dont want).
~                                                        
