from sklearn import metrics


def evaluate_cluster(Data, labels):
    sillhouette_score = metrics.silhouette_score(Data, labels, metric='euclidean')
    calinski_harabasz = metrics.calinski_harabasz_score(Data, labels)
    daives_bouldn_score = metrics.davies_bouldin_score(Data, labels)

    return {"sillhouette_score": sillhouette_score,
            "CH_score": calinski_harabasz,
            "DB_score": daives_bouldn_score}