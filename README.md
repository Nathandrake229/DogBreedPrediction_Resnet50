# DogBreedPrediction_Resnet50

This is Dog Breed Prediction model using Resnet50

the model is in jupyter notebook
for the model i used pre trained and fine tuned that model according to the dog breed dataset
I opted for pre trained weights instead of training from scratch since the training dataset is small for a Neural Network this big 
the model was trained and saved in h5 format which is uploaded in the repository
Since the weights .h5 file size is more than 100 MB so i used Git LFS system to upload the file and can be downloaded separately for testing the accuracy of the model

##########################################################################################################################

# API around Dog Breed Prediction
As for the REST API, the file "api.py" can be deployed and tested. I tested the API using Postman locally. The API accecpts the JSON data in the format {"image" : Base64 encoded image string }
and it returns {"breed":breed, "score":prob} 

here is an example

I used the website https://www.base64-image.de/ to convert  ![alt text](https://github.com/Nathandrake229/DogBreedPrediction_Resnet50/blob/master/dog.jpg?raw=true)
into Base64 encoded image string and then sent it to the API and got the resulted in the prescribed format
![alt text](https://github.com/Nathandrake229/DogBreedPrediction_Resnet50/blob/master/API_Screenshot.png?raw=true)

the base64 encoded image string for a dog breed image 

{
"image":"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMTEhUTExMWFhUXGR0aGRgYGB0bGxggHR0dGhgdGx0YHSggGRolHRkaIjEhJSkrLi4uGB8zODMtNygtLisBCgoKDg0OGxAQGy0lICUtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAMIBAwMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAAAAAEBQIDBgABB//EAD0QAAECAwYEBAUCBgICAgMAAAECEQADIQQFEjFBURMiYXGBkaGxBjLB0fBC4RQjUmJy8TOyFYJDwgcWJP/EABkBAAMBAQEAAAAAAAAAAAAAAAABAgMEBf/EAC0RAAICAgIBAgUCBwEAAAAAAAABAhEDIRIxQQRREyJhkaEUUjJCgbHB4fAj/9oADAMBAAIRAxEAPwDE3hJxoExICMJY1y2iy4VYFpxKSanE6m993EaSx2FIAMzApZbCgmrZtQe8U3pYUEqnFKEqQapYUHTQ7xw3GUeJNFf8RiARhUTmzhmJ0r+PDm5xMs9kmklg+JLBwHNR4v4RlLReKBM/lqAYZgMOppV4eWW/k8qElZNXFAgjdTu5pEwbT2gXYmmTRimYpi5eE0LZv8oLbtn0g6XMRMQCVYj8pU/iwG0G2O4lzkhcwAoquocHZ3zDe8NLvupEwFQASArIZeEdEMSauQ6EsmSkS8BJoe4bYfaCFS04MKWBIIdTlPj1bSDbwugS3LOBl4xgfiGVPKggF06AP5Eawn6eLfYcTVWVZEzFKmBWAAhmQl9m+saZcidMkiYDzjEFJCmNRm+2VI+UWW0iQkHGTljSQwf+0g0I8I+m2G3LmS0GWnipWwXXmCdAW0bXtDSlj66fuTNC697pUizpISUlsag/MPEZikD3SBNXiVzhKTRxynoTV2OZh/fNstBUmWMBQxThQXUkZAEkVLGBJFjkIThKFJJYOgVYZ0OXUxj6iKi7RMd9niLPML4EBII+VWVd67NWIyLHL4pQlXEUEgF29D0iy/ky0y0oBUSQ4UmhIG/tSA7JNl2cOpGNanwFRwt/l9olfDj11+Q2NbrCrOocbCUYqKWAWo1NEu7RRaZ8jimYlTAOClgcTjMDSE61TFoM0rISCwrSmjGoFWeDLFciZiAUEmY9VNyEmuEbNFfGfXhdBxI2azyyklMxKAogsxJBHbKgEHSrKFTELTOKp2aSSabhSdE+sRRY5lmWFlGIgs3zDFX5tAW2hmgiUgz1tjLgOGZ6sRqI55Tqyki6bY5uIHjAoJZRQwIJpynvrA1qsNrJCVjFKyzcsKYleEci1KKf5qkZKVhAIoKij0i6x2yZLllSl4TQpBBZjp0ETH1DUeLWinBN2KVSJxKeEkGWCEgjrmN30eGhMw2rAuQwSl0KFE7OVfSCLJfZ5mABZ6epbQP7wlt14TlBWIKS5oCaB9+hjOfF9Iq2O56J0oYkoEwkuWNUjensIUTVLmzxNRTCGmM9A+XV2jpUq0EKMrGlgwZTu1C76PF12Wcyiccw41hl5AJao/NYV0hBVhlywoY0FIQ5SavXd4ovmeZikISA6iFOoZECgbXaJy0licaSlLYXer1NfXxgH/yipigpOFSxROIEN4imUKN+RgF6WSbxCQGUEAmjCtDlQaQFbbDPXMlrmpoFAHCzpHcBkpjS2y+eHIUUy12hYYTDhA4ZGTqpTpGesvxSnns85KgFs2A6u7E9ekdWNRtSX9foZu72ZSav/wDpmYXclhQOWMMbTLTMxJM6aolwQSOUZoBAGb5tDK87DLncLAsItHMFKVUkCoKgkDm0d4XItv8ADrSoMuYlRxKQXyLAlwdn8RHXOL5KvuTd7FMqTMMiUOIxSVMgFixJcn+4kQfd09UlPNNmAKoADR6Gr9Y62qlTZi5stK0Y1DNiKlyelXyGsFSpMucShqiuIMxbvl36Rnly068D5FC7slrOMLWH3ANdatu8dFX8FhoqclJ1Dqp5Fo6FzX7v++w7YTaZxSgrSp1hmU1dwHGvQwTZ7FNtEtE5c+XhUcJB/SdiNX3hHLnqRyJLOTnm+veCbJNBC1KWy0KDJA+YfqY6douNR7QOweZYlyZmCakMTtvkRs40i1AmyOJJSwC0DEWehy8dHhtb/wDkDq4sopBOE1KmoCenSKbLZypZJVgLJKg1MOY5tunWHvloEaz4Xtcw2PAoHQJ0pr4ZQB8TqmgpTJCkpAFQ+euXeCZV9pACUB2LUHt0hva7VgTiKgDmABl3epjfpGiEmJZlfzEzCBqfSmfjGSv+3ABeEEEYSku9fGNDfV6pmDECUKfU8vhtCWTd5mkiZUEFwDAhsydyThiIWnGFH5Tr4bk6x9qumwyJKUT5LywEgFCTQuKpIOTdNY+dWu5JXBKpeJM1DMn36v1eHXw7ek9SBJIcHOmfTaMvURbjaINOLxlrQCZaMJUSasxfOlXhgu2SkhQUhJDM4zKTkX0jMptwStCZic3AATWh1b0jTWW6uULWSnECCkBlAdTpHBTS2K7B5ljcFMtJKmcYiFYdUvtGQtlmVxQZxSkJbGl6MDuNDG7nyU/LJZJZsTmtNWzjH22418RImlSRMLEsX5cnZ6PrCiDRCZbUFZxpwIVkAOu20ObNesslMmWwA3DPTOsZW97omowrBBlgtjDnXbcbxWmRPs0wGYlRQoNiFQBq9M6wODqxXRvJd74zgSEBKB8z02YamkIU2uZxQkqMzCpgCDkTmPaEki9+GqYEjED0yfKrZRobE87AuavAkBsJ+Y1d0tpEuLS2Psbm1SkkCUkIOLDX5icy75iJ3hY1LlKQJlCXJGfRngabdyZczEVKJUHQCxJf7bwBKnqdcuZaA7hnY7uzaZZxGyroOnXUmWAZUxSz/wDJiIBO5BpTpEJFjlWksqasF2PKwyyB+8V2LDLW6gVOGDGj5UPiYacUZrdASQyQGcjV/wBRiubdWwPP4VawkSZ1Ucp0CWoe8VKCC8tS0kTaAqdwoZFzpqIR3vb1CZyEc1AtKsgKkED9UHWe9EKSiWcIIAGNTKYnWmukXODrkCLr8RMEqXLQOUUUQp8TasA8ZmzzcSwGILtRterUjTi1NMXKIAeimpipylPu/SArNYbGtVSpM1ziYmragOz9YzkqexVZRfFntKLMZaJwmS1EA4AMZzKj1agjHWCycVSpSUrmTnHDOJgGLnGD0p4x9CvSUEFCkJCGolSc001OTnODLJYzi4wKVKIqEpZXQqIzjXFlUW9CafRmr2ueZLRKEkPNKwF05HVQVOSYYIufgIUjgJmKU5UrMJp+nVgI0V3TpawsLl4Cqhc/Nua5VhXeFz2hS1Kkz2SPlw6gs6TsdaRtHLyVSf2E0/BgbRZ5pCQAGDuwqNRi3YQsUhQGJByOFxRzGu+LLOhCGS4KT85Jdav1EnXtGVnrZlPy/MWBAfYDqYMfzK0FFk+65qSxmo0OR1D/AFjoul3+GHInLVJ+kdDrJ7fgRO1XYoh0fMGKAk4qbmKbCgyzzJxLWda026GsbC7lyS7goOTfvtFCkS0FRSFBKqrOru1PSJeXVUW0ILUMCBiSQUvTNwe0VWK85KJWBSwC7lOFRUrq+TDKG00AEqlLKyH/AOQaHOmphXb7AAsOBpzJBDk5Jpt1jXDkW2xL3NxY7lHCllQwlacVCHD5AvAdvsylnDjAIoHLP2jOJn28y0hKuIDQukoCNqq18dIHtInAoTNWVYf1D2eOiTNUMhc6lKbDjTkQGqPvB112NNnK1IxqAZISfmSToRQwuTf8tDgkhbeEUTZs1fPxaYg2LbPPUecNdCHt4SVKOINQEFyPJ3rCiyz8NaONiQAIMFvSxwNMLfMQw8ooXZyUlSgcR/tIA7BqxXapiZtrEZCpcpayUL+YHM0ycswfaPbVPVjeXO4iVCowuTow7ZmM7dF6yuGZSyQw/Vmeg84bXLLTLQTLIVqUkkqb6GPLyRlGXWiSqySCVKL8NROFKQ6n0f8AYQdetkmKHCBKl8pJFAH6UKvpACbczr4YSCsA4qdm6ZvGgXb0Mz4lgbgM+x6RKoBULswUC0lgxS7AMdq7PHXlNWyVAEpAwsf1PqB/S0B2ybhdSlAgjCQNBVupPWJyLUiYAmWqiahIep3rkYTEZO+JATNwy2TxM0OGBTlXaCrAlKZaVY1cfLmIZNaApo9KxpLNd8qeFzZiFpSkhCWo2hUkmhDmPJV0yArAsEkHBiHzKBqkrJyLDPpD7VMSsAu6YuYopUQqaxCD+mnVqRP/AMRiLrWOV6JIzyOkFKssuS/Bdg4U5Z6vnt2hdeRKinhipdZxHNyzApzbrDS3SKv3ALxniWSmUvFiUBkXSkDM0316xSu814ihSipAfFX2MUW608CYtCqhI5iC5L5EbDRukK136hubEMRBJw5jLON3hT/lsSDVTBIUmomYlAkM4Z8wTkqDLVOQiacAMvJVfNm1z33gCXPsyiTiVgokkVamUXWlaFKKXNAAgqofE5V6Qpxa6G0aSzWex4FK4yipYLYkmij/AEvC6QOIp0Aky2BZgok9NYh8K2ZCk8Vc0LwHCuUXomrqfXcNC7+NWm0HA6JYKXGpSe9SCADFTxNRTbJ5bNjZ58+aEpZBxqIHMRUUNCPWB18aSpRYIAzGJlHYbHwjxNqQFhUtakqIfGlPygqAViAGoGcM7bJM1lGY6U0xHXowoPGMM3BRTj/r7DVt7Fdu+JlBCDNQ7kKdIcAZEdSSIayL3JDAKQgVAAAJf+07EsT0gS87BJIlgqZbgJFMLP8AlYPE+VNWqTOQjGlmwlgRk4OnaM011+Slfkw96YeJgWozVlaiUDRyCCWBAfKEsy7LTaFKlWeSpkHmw9ahydto+nWC4LJZ+IUllrBDlTuDVg+zZwHZCbI6pCiTMqzUVm3iHjaOZRdRHRlrr+KP4eUmQZQeW4Ls7uXzD5x0bxM8qDrsrqOZKA5jo3+Ll9n+TPhEwnHwgLokM4S2QOVXzjxFqEwKCmrkl2Zmy6mM9e+Jcwjmp8pNHGlMoY3VYJhTxZqTysE7EndquNozlBJW2AzZihKklRJoEM5bR4svq5LQVFASFAkUKmILDIvXv0iV32BRtCVqIAUsKFHbeGF7THmEpmqJUWCKP4GvWIpJWDVIX2WbaUAy5iVKYkGoNNK6uYV3ySmpcJO8aW6kzVFLSlKYBSyqiQHNOlG3in48lKXLQpSGY4QDQb03GcaYcnz012UnoyAupc1Uvh83EG+VWY7RtrNco/h+GUh0OS51J0OsLrgtqJSEsBiA5jBV+XnxEpRKJ4hIoMm6/SO1lIttE5MqWELo45fwZQltFpS3ICehBUPN29IKmCYpKUzEqZNMQY/62i5NjlpSGqdXDHx/aGgFMpQKqpZ+49YkL5mSXKeYUBfxbxi1dmS5IS3UHPwMBzgC4UGTnnro0EkpKmFB91XsmaFJmGpfOnZjqYdzUA4VGbiLB3zDHQChH2jD2iYQw8jEjaGCVEkVjmn6WLdp0TRvpdnVMwPhwvmc+hbNoKu+xyZa1LKaA81TSnLTqamMGLepICkqO5rB4vyYpJxOU9aOYzfpZLphRtJNoUmWJuMBDlKE/wBRdx4VgedaSy2cnNZzYCgd4yM2/FioSkB6Byw7PlEzf6lp+UAHPPd994n9NkDZogvipwhKllTAE5Nr80WXndaiooEzNIAUn9GhbU0hFJ+I5jgBIDflRtEl35MCsWFGQpVqZZGBenyL2Ci+VdAVLn2aXIUZhAUJiinEvC5DFWxLMNIV2H4cnSUJK0JUSopKHxKCTTJoJmfERXhKgxSX5d9D26RVbrzxkrllcpShzEcwO53AbQRu5ZXFLjT/AMEpNGUtgSJpGFfzMOUAgPTLakOrVZxiClqGIJcpFSRorYNk0NbLIlKlIQpTlD/zBQ1LgMXJ7mFF5olBaiEqCW/UTVtO5NfCIlk8UOwm5kEzv5UuZMlkAlmC2GbbntAl/WoG1OConCg4VBgS1E0AbD2iVxYuKORRl4wtRSTxEJDvgD60PhDb49sql3jKQGJXKlqACcO7FQcsd6x0Yor4ZCdSLLqlBSn58JGFQVodeoD6GHFrIw0WMJHN/adG3P2hkv4PmSJQmKXjwgmYHqs5ihpQ+cKZuGZLJM4NLOLCE0cty94xzYE3bHzvo9sFqlIIC0HGScCi4xjp1eCZ9kZcxYWoKwOQvPcDKp2g7+EsiykqnEhKAUqUoAJq5YNRmZ4QXpecrjErXiAcIIdlkZvuOojHNg4477HjlboItFtKUJKEqXTEXGRb3iuTxpif5KBiQAVKJfDiqzDNzrpHtjmy1HCHLpemj7gtSGV2hRfhFgQMaXSKpyLpfCM4n0+OPJJqzSeo2i67ETRKRxpajMbmOPPbXJmjod2WVNUhJeUKUAUVBhQV1pHkep8Ofj+xzcl7nyVchJWpKzhUWeoKUluZiKAPXpDSx2VcuUyVKW4dIrkXL+5eFdlkArWOHjSM5jthJq5A66QQLyXLX+o0phV8qcmA3oax5krejZjqX/EJAA4fEYc4rhGo6KaB7WEAEsvGguCnTQGsV3Hak4iogsXdqEgDMjeLETCmziaSCFzCObRIoA/WMmhSegy5byfGorILVSdSKEkZPlCf4ptBnLl87kOCdOlOzvDKxWLiJwuAACSWql60hZa04QgpNBR2rG/pdZOS8CStUWTboSlFFk9gAB4mGFiuZNnJo6qEF3JoSR4N6wLLSJxBWrlRUDMqO57NDOwJxGWxfNRJ6gJA9o7WzZBFoICSUhiK/Q/vC20KcBaQ+41BFKdYsvu0qySoBXuxoOrsfKF67Uqa6paCktzDV+m4gAGnzS7FGHxqfA/SAjLBUSCQE1yy8zDT+CWUgkk1yVQiBp0mWpwRzDz6ONYYhTiClNp+ZwPMswJwh889vOD7wnYKJDwAJ9DSure8AF8kuSnYNFyXA5iIG4uEOkgb0gSbbFmugoYADlTDWuWn7R0hQUWaF8tYWQAIISrCvNmgAbCW1dYqmTSqhFBAK7U6mentA1unBBAJodvrAAz42QGm8XCz1JhfZ7agkDRmLft1gyZPGUAEgsJyNNc4rnWkjQN1MSlYWcvzZZn/AFA9sUqhanUwwNTcN6pAShaU4csnoS5rnFnxHIXNvVNokoxy1IS5BcAgMeoyEY/+LpQQxue9yDRgabsDoSNozUXC3D7EOJ9b+LpKJtmmTFzCBLDpQSAAW11OfpGGsl1LTLXiVilTBgTMUlQwkGhTSrjKDLLblIHLhmFbFQLEq1JSCWeNXcvxSibKKlggyyWfpq2ghwywyveqMXFroyfw9dhnKnWYTUpCHBcAkgVATiqQfpCO320iYpKlgFIKQEpTidOTH9IpDO12C8ZU02nhS18ZRCElR3JQpQFDrR9YyNu+IccwvJlYf7pYCnyJcNrBKNJJKjWHZ1hVMmqdSiwDFRNC5pUZDrGm+FpE2yFUwHjJBPK7gaVPUHTaM/8A+S/kmWqXhlhLuC+ImNNdloRLkpGJVEghVU/Nk7PiyZ+kYJzhLlFDnFPRTbL4WVqMyZaULeqUpICdgADkzR0QtFtkqUSJyA7GqFnQPVqx0V+syfs/Jn8IH/8AIplhcsDnchThgasK61pEZVnKlS6kKTiLEVOLLC1e8ezLlxzlqUoHmKgkbGqXOhqYNs6z8qUkKGp0GXmY4ZU3o1lt0V2QqBUpQS4LHtt1ic6XilpwkYCXSk1w99wd4tMnHVZKQNcgo9fzSB59mxhYTMyCcBNKagtq8Q3ZDTPJk9UtQYgFXIEpB5iaZ6Ui+9rMJCUomFlM51Aeu0Ze1TZ6VlRFJTBKtHILNSpIdu0O5FvNoUrGStRCQD1cbeMdOCDjJFR12CptSUqdLsOjP5xdLvFSDShZz0Gj7Vc+IjZWP4SS3yYjoNO5IhHM+El8TAogB3U2QD6ns0dxoZq2TlrUg9aNrqVdv3jaXakJlJUEhRFFaHv394utl3ykhwAEiWrC2dAXPnhHjDO77MBJBZjQOMlA5HuKwACWhAWBTNxGQvawYF4hkWHZi4jY2yUUg7g/jRkrxtHKcXXwOkAGatqDjoaRWJfK9cXb7xG12xyw2f8AOkU2q8MNMi0UIAtE5RLDSsQM/lb892jwjEpxV4mZLOTAIusqSA+sQtFo5iaxZK2ETNj1oe0IYCu0lVaj2jpcpSyK16wbNkABgIqQT2MFgOeFhADCg2ET4SVB606ROTKcCgJpXU7iKrZaQkOHCTT7wgKRaRiocoPXIEwOCxaEpsiwywygQ438oYXZaXp4GGAvtMmrB4vkpCUmrEdIbS7IAaj7wVaPhdS0FSCH7wwEdutCkiSpD4mz1HMW9EiNn8Jzpq0cB0qAbEpVEjEaj+4P5wgtFxLKsCiUslAJ7JGIj1j1NrlotHDQspSgAOQTibU9HjB6domSPovxX8Qrk2WXKnSknivLStBbCR8qhtvTaPmHxBYC4HDMvCkkVJKhRi36Y06L7VbZ0mzrlcksuS9KCh8do1yrvQHZP6QmtWAyismW2LFDR8XlWRUzChBNThc0ZzTF0jYXNJXIC7OJRUhRGOapzhANUhjkNG1MaK8bBKSSoCqgEkUAIBzPaAZd4lUxcmYoAIUcIBLLIDirVyrGMs8ov5SpRXk8sMhKpaVFc+WSPkSkEJ2bEHyr4x7Fy5hd8TPXMjOuUexf62Hsyfhiu+18MBMs8yjUpzU6QWA0qYquyeUSMSlKSozMNKFbd8hWp6do1cy6ZQZSvmASrxYANCm9bvTMlkKqDQNQjM6ZDLqY4YqL0y+GxbOmoWQkqUDq2WRNOlM4ZWOzDhpJqskgJOjNzOPKCbHZBLSoOkpQHIbSgwnwHrBKE4pzgCjMNC4evgTEyjXQcPIqvG5ETUibNWvHQ0VlWp6kigOkLLllykKUZOIJCgTidRDKAJUcg9aCHt4oM1eEFkILKLZ4Q7U0YmPLsI4BCUYEFZGVSKEvsKekXhySjJWNqz6Ndc5IkO9T7xG02NK5TnM7RjEXwOIiVi5XctswNO7xsZNoCklqJYMNo9FgjHXjZgZcwqAAHKkbJo7dYZ3LMeUEqL0B77+dfGB78TimBLMmnk9fzrE5wwpLZYnHY5iGugKL0mDNPysX3GcfN/iOYspUKDV/GN9/Ej1eMh8blGAlgHzaGgZikKYE7sIGKSs9vWsTSHDdHjyXOag/eKEevhLCIB3Of53ieIE1rE0KArCAjLbWCf4gCggeaRvFK1BwIAD+OSGbtDS67oxMpX5+PCKWSS22sMk3spJAdmyEIDVSbCAkg6N6OTC6+LDioNB511hdKvxQYex9/L1g2yXmFO/nCsYstls4akAgijFs8/akHXfJSpRUAGfdiPKLVWeWqYkrS7DKHFlulCapGEnQedd4YBFns4NfXWH1is/Ix18D47wFYZLA4hqMof2eWw6aeETY0jNXzLUC9GL5a9DGXuu4yStcxWFClAGvM2JyEjNywA7xurTKop9d6iEyLrQpUuZxFIUglWmHZzq+XkYzyOoiYBYLfMsllWUpBXMqVUxPiCWP6gkBJZ9SY0tgt3FkoK5qgthiKThAKhiAbsRAtpsKMDBbpmL5jhcgmvhp6wDbrtRMmIRKnlRYkS8nYlTd65nJmjGLTWyl0HXnYJYSAFKRi+bETzDU1pijKWycuXiSVABFUHCXVXCD3rD28BMtAAwJKZTEnE+yasMzt0jp9gVMQonCGVy5kgJqKH5XzjKcqlsl7ZG5wVyUKWrmLu6S+ZG8dDK67ODKSSUglyQ5VVy9QM/aOhObstQ+hqZskKDEZBNd+WKf4ZLaAKNaRbiDF9h7RAzHSS3oYyuwPVWZISqgrn5j7R7KsoSxB0j0JdIAP7kx6oGvSg9vrDsCqZISrEWq57F83iiXYQmgAIdzsTX0i6Up22yAi9LYmz19/tBdisyF4WYomFaQ+ElJ2AJxD0LRrfh+ecRlKPMWXU5jL6esLJ9lBWQrKYH8U09iPKMfinlTIU6pZoSdBQCnl5R34Z8lRJtr5nNMbLOsVC1uj1jDf/tomPLnJKVgmp079/rDm57W6QNyw7CpjZAHTpyammWUYr4ntQUkpLbONR9xGnvWzcNy/bsa1G0fPviOUsgKBJBJo3tFITAxIo4L6faIWiyYWLwPJWoB49WFnP1gAuC0tSkUrjpdmUcq/wC2hjYbgnzVBKUGpzOVYlyS7YCriw6uP4bm2mYnlISak6ARrri+ABLdU+qtE7beMbawSRLQEpSEhtI5snqUtRCjC3xcMmyyuWqt9qR8/mA4ico+p/F1kUtBLmgcn6Rivh65eOtlFkDM/brGuCVxti8iGUgk5w0lEyzzpIj6ddPw9ZpJLIc7nQfjwJ8S3XLIISA+2raxbki1FmZuYhaSdXAHn9o0ljkk59W9M/OMddi1SZhSQw+YDxqI2txTOIo7Ub38YYhpLnBClJNKP5U+0FS7Wnhmr/7aFltsfFnSxqnXcHOKES1S0TA7kL8CB/vTaJkUg23zwZZIzgO5ZDhb1oadP9tAE22goL5uz6PDj4XS4B1ypq/56RnmdQF5HNikoSgAsAAFMak+DaQKbvSFiawSUgpSEhgE/qbqQRBypYxksXCde0VrkklRJoGA9z+dI4uY2QsmDC0pgCQSAwyDM+uUTmyhMd046ADox9WiyzSTlifps9Q/hEypLYQVULn6dxAJItsMpCUBKQGDjLqX9Y6CrMsBIDjX3jozbZpyBJoBY00PXI+OccUuhWmTeTn1gXAFM+jFxnrn06QUFcrEuSaHz+8BFkbMn5fP88YnMW5YeBikLo/SOJIelNIAR7Z0s/QvEkr5khxoYmksFE6n0aKFlg5DOfIAU9YdAV2sUC/6FOR0FFejxnZctuK2QUfUn7xo+IKwjsqGM0Gof00jp9K+0LyfPL7uoqUqlaeLxubsulMkIZSiAkHmZwfz2hV8RWtPHSkDZ41C5YMtBOqX+0dMpNTihCq02l8RLEJfOM9dF1KtiyjDRJ1P9R0G9OkPbdZnQXDvt+ZxH4bm8FYUxagIDB/3isjfF0Avl/ABGMOGyHfXyYwzk/BEoITiNa13OXpGmVN5C1Coa9f9wVwzhDnR/WvpHnvLJ+Q0ILJ8LSEsAlwH/avh6w2s1nQkcqchl2idnTVR2yG509IJlIAcmtfoT9ozbb7KKZjlaiVODQdGDfT1ipIbESwoa/t2iqZOXiyAABdhkK17/eJzCrC419Mv3iWBVeliBlhJrR38BGSTYOE6RQ5iNraZhUMtoRXrdylOdR+0dODKk68Copuu1kirOMjvF1rnBaDSo1gawz0F0thI094PUgAfK6SKx2MqLMHarLjXsQaQ7sWKSXzA9dYhecpIYpNSqj+GvnGmtNkEyWlhXSnSMsmXg0S+yi7JxVNKzQMa+J9YtnpJcKyzxDcvXyZ4XyZpQCgiv4INmyinmUpkOyW6CuemnnGnJOmCQntNmHyiiiajTvDe4ZfDLMx6anSm8L5lokmhfEXHMWbJmbSLbmtLKYl2yOohZlygxeTWlRcg9B7R02WSA7jEf9e8eLUAkqz0b2j20Ly6D2/1HndItnspFFeHs0ViWK4dA/aJyVjm3I9HH1iUhDAAtzCvnSDsmiPD2Zs9dax7Ep9rKThGGjeznPrHQrRdIolginR6axYZlQNKmmf+opnTBjAcZHuwb3pWPMShUswz7M5hWZpnqZnM2HLy3eLsTnJz76wJoWOeX+49lTSHUS2iX9/M+sHgdhMpQYueX8+sRnqBDZ9NM48tE0AB+hKQK6D3imXMGWT1bXTMmHdCJolAE1zJPhSFdoDKW2qm9oYlb55ZeZhcBzHuSH8qx0+l/iYzJCSV25IP9xrXLIRuJiQQhjQE/URj7KAbceYBTMkHf67+EbBAoKjNq/bxjTI//aKEATpTgv4CFl1qdbYHZ36bQyvpZlS5igCTt4ekCXDJJQ5FTUkGjRrmfyMGOzOLJIcDECd2/DBo1J1GpygWWhKmxOQGLb0LPFhJAGVGIfLIM/nHl2JBSd8g/jHJqjx9toglThL7/aPSvIa1P3eHZb0QAJSHoSAS5/HrHLNQlqMKe9POscSXxaOzfn5WJIBxElgPf8PtFAjkkZa76DrEZ8tkuaufNtYms/yn6gRQsunqKeFf3gbASX3dzfzEDu35vFF1WwLLLNB+GH9oAKCnT8/PCEV53OWCpYZVKvT8eOjDnrTBgd7JC+UZAHCeojQ3HMK5KGqQGNcmBDN+aRmZVmKXBLjN/wCk6/nWG91zDJXiAJSosoDTdXgPSN8sOUQ7Gl5XWlSwo0wl6fqYO3nCK8pikh1mjs31HTbvGqFpxDECCmmE9N4CvOwomIbUsw8HrtlHLDJwYMw5kGbMSCrCgAlSjkkfUkZCCgtImcSVJWJUsAEOVYtn/ppU107RK8rgmIcqW0sBxRz/AHYq0AY1OwaFgty+EpCZjSgxSMiqtSpsyfTKOxSU+heD6FYLUiahCkl01c9dj1gqagEA7B/Ut7+kfKLqvGdZ1qVLVRsSku6VfUHIRqJ//wCQZaWTMlKdSEl0ENUP+ojJ45p+naetis1kqVUdvSJocnwZ/rGfuP4ts9pmcJBUlZBwhYZ+gIObRouE4ocsyPz8rGLg06Y0WYwcwCdzHQPwusdE2MW2xXMlI5a1PR6xJU4JfR8tyGJr1gaejlBSvUmuQ2FC2vakcEFWxIBpQZZaZPGbTJQbOJKmAeorkHNVV6CLxUBgQQMtQ5fzd4rLFLkktQaDrQaftEZMwMXLnr1+0V0O7LlaOSzl3zbR/WK0pIBq/wCZ+EXy1YkO1Px89q0iEoPU5fmcOgIrDqFOUMNidH9YBsaE6lmKmO7EiGS5ZJLaDy08coU2eWFSwknJta0Ne+cdfpu2DEhutM9doAOBSWKFpzSsF0kbh8xs8PbJeYWtaCkpVLIdJ6pBcEUP7RCSGnLODC7PWh2PTtE5K0mcF4ebDSjMAWru/wCZw5O8tCLPiNKTJUf7XgG45QZBHZVcwaU2g/4h/wCNWx9KNCW77eEAIUXocNGzoH2qKxrnv4ehM0yMyDRPTt+CLCzCmdW9Yztl+IEE4XKiaV0ctXz0hlaLyCAAKuGHqSS35WPNpodoJnzcKwHNQO1P3PtF00ZJevbR2HrABU+CY+LDk2rsKeY8niSLafnALk4Egjx/7PDKYbMSAA+p3z18o60zwA/Sp7HKvaPJpCgCaUyObs/7Qltk9kFGOpOEkbDNuuQpqYAvyMpNtCkIIriOIAaBgPTFnBCgxJq9HA7U9oT3MlSpKFk1BwnqCQQOmQh4Vhm1Km7M+/V4T7BbKJRBSreufq0ehYpR2ow07xWunQE1Op1p61/BQFEpwywwqXbLNyGzLPAKgWdJSkBSnCVlh0zAJ7xZY0jAqjFTCtK/6B9IMtRSpK3AISdQ9a0A9Y6zSMWAHlcEqGVWy6ZFXhHZiyfLxZUTyzkS8KC2E5ebAds4mqanGokslIKUjUqzLblm9YX25JXLNWx4kpYkYUJ1fMPk+dabwVZ5Z4aVroQVs/g6uwOQ7axjkilsUlsMcTELSciClXkKfvGRt9xpl0FUqr2IDgeLRo0LShgmrEJH9xOnWgcnp1i+ZJSUEKq5zf27loWLI4SEkZVNzIVISKOqYUYtaMRXSpA8Iy96XPxJqQnEkqQhxsAkJ9x6GN7PTgsqUsSQskvRWIa8rAim0DWCxfznXUhAB6AMyW3pU7k7R1PK9tDa0Zm5fgNSVJmGcUqSujDNmI84+oSVU71J7D94EEtIUSaimejagbxSucGBFQHHStI5Z5ZS7JohaUTsRwrYacoOnePYNFnVSmgyIjoniMCVZkpSyRUBgPYbbExxlJANHCjXegBI7faIImnInKnizQTNmB66berRnYJFYmVIGlGagdhTyIjyUnA1GJOXhl5CPMTEnJ8/M0/Nok+Jm/Ke0OxUz1a2JY/KHPcn9susEylPR6EZNp40ygVNMT5mhpnm/qBBFnVUDQBm6Vir2NaOXOYYabk+bU0FDGbtOGVKSQSSlGPu9T6AUjQmUCcJ2qWyGr+ED2mzJKuVNAhg+lWSOp1PYRpiycbDbFUtZDqdjtswcv223EMLM7pQQBiSMXRh9VHLoYjJsf8AKVhSQ6yG1qRiL6Fya9totXLHEqHc+AAf9/Mwub5chUwO97aysLEp16CgPf5jAl43SoqlEB8LppqGWQf+vioQXbrGVIISHUNNsWXg7eUMUpNHPy6Dd6eHL6x0Ty60DR88utQXPRKqDjUSMgySQVAnRgWjYzpWAEqJK3DgZcyGNWZnbKLkXaEqK2qU4RTId+7nxg8yyovhIdJ0/uJT9/GOec+T0OMaFtkQpI5mGbAlmdNKnYkeQi+1zCACAKJoRkDvXKmRMEzrGFqD5JzrU5Go1D18o8lSDiAowZg2bUDxkVQAhCwFB3VgYHZgBQ55nPpBpuoK51Hw6mpfxCfKLZsrN8ywfVtXgierlD1AzfcJfzirFXgAkS0S00oSsEvqw9gPKPTKAWC5HNU9A+Q2+0XTbOkhsm8tj7wVZGy2BHnmB0zhL6gkCTLUk4t0pBAPUGtd+u3SK+H/ACsCSoLUKKGbqISW0JZ4JtdkDEu5+YDIBmp6N5xHhEFxWgZtBWp6l/SHY7K1oxKSkEMkuXrXWu7FvGCStHMFUUaMMwDtsWSoRUVMklOZU2WpbziK5fKA5CnJUTVzplpX1gjKhxdAyluuW7AlTajCAX5RqGGdBF9ptA4cohwnGpm3fD5ODEESOGSpyaa5CmlN4Yz0kS0ADJIo2TkwOVoFsQ8NAUkAlziYltSxV5ZdBo0H2i1YVBNQluUNQjIHPKob94iiSAKpGJm8NvXyHWJqlEqQGACHHgBTucvKBO0JdnTZiXxguKh86vX6wvuq7wmZMZRdSqk7mqn2huJYGTsBTqSXiiRJwncqevclT/m0Up/K0IIWlwQWqO9Tt4PHWYCnKxyqX+Xq3eB7MvEVU1LfTs/tFi1Mls3Om+flEpqhkZ0/mNVDsI8i2WQQ+B/ER0HH6hoWIPMf8j/2gm0f8g/9vpHR0YxF4LLWPm8frE5Cj/L/AMfrHR0Uuyn2ePX8/rMeyvn8/aOjorwKXaJjM/5D/qYikVV2/wDrHR0NAeyvkV/kr3iEz5j/AI/VMdHQDJyRy+KPcxfJFT4R0dFMD1aRhTTR/aISVHiKD/jCPI6F7CfZTKyPdX1gmSOZPcR0dCY2QRl+bxSnI9vrHR0SgOlf8av8vvFg/wCQf5K9hHR0C6QMv/q/NDEAOV9aVjyOi2T5Kj/8P+Z9lQWsc56D3IeOjoSGuge0/Irt9RBM4c//AKp946OhLyUhfL0/yMFTsj3HtHR0KIvIJZjyf+31VE5nzgaYTHR0Jh5LEDkV2P8A1ERnHl/N46OihLoHsyBhyGZ9zHR0dFoyZ//Z"
}

the return data is

{
    "breed": "scottish_deerhound",
    "score": 0.9999816417694092
}


