I had to make a very slight change to 'tester.py' to make feature selection, i couldn't find a way to send the selected featue to the
'test_classifier' function without changing it's layout.

I just added this line:
'features = SelectKBest(k=5).fit_transform(features, labels)'

after line 30 ('labels, features = targetFeatureSplit(data)')

This is the only change i made to tester.py

You can use my attached version, or you can modify yours.

Thanks