from sklearn import tree, __all__, neighbors

def DecisionTree(train_pattern,train_label,test_pattern):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_pattern, train_label)
    result = clf.predict(test_pattern)
    return result

