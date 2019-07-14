#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:37:06 2019

@author: mloucks
"""

from urllib import parse
import urllib.request
from urllib.error import HTTPError, URLError
import time
import os
import socket

socket.setdefaulttimeout(3)


def clean_urls(geturls, badurls):

    fr = open(geturls, 'r')

    urls = []

    for url in fr:
        if not(url in badurls):   
            urls.append(url.strip())
    
    fr.close()
    
    fw = open(geturls, 'w')
    
    fw.write("\n".join(urls))
    
    fw.close()
    
    


def download_files(geturls, location):
    """
    Download files into a folder given a file full of links
    """
    
    start = time.time()
    
    url_lst = list(filter(None, open(geturls).read().split('\n')))
    bad_urls = []
    path_lst = os.listdir(location)
    
    for url in url_lst: 
        try:
            path = parse.urlsplit(url).path.replace('/', '')
            if not(path in path_lst):
                urllib.request.urlretrieve(url, location + path)
                print("Downloaded ", path)
                
        except HTTPError:
            bad_urls.append(url)
        except URLError:
            bad_urls.append(url)
        except socket.timeout:
            print("Timeout, skipping")
            bad_urls.append(url)
        except KeyboardInterrupt:
            print("cleaning urls:", bad_urls)
            clean_urls(geturls, bad_urls)
            raise KeyboardInterrupt()
        except Exception as e:
            # TODO: add other exception cases
            print(e)
            continue
            

    clean_urls(geturls, bad_urls)
    print("BAD", bad_urls)
    print("-"*30)
    print("Download completed in", (time.time() - start), "seconds")
    
    
    
def main():
    #download_files...
    
if __name__ == "__main__":
    main()

            
        