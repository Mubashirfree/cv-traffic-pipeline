import csv, json
rows = list(csv.DictReader(open("eval/clip_gt_from_xml.csv")))
R = [dict(c=r["clip_id"], night=r["camera_id"] == "MVI_40992", bus=float(r["bus_med"]), tot=float(r["total_med"])) for r in rows]
S = [
 ("Q_multi_bus", "multiple buses on the road", "object", "bus_med >= 2", lambda r: r["bus"] >= 2),
 ("Q_bus", "a bus driving on the road", "object", "bus_med >= 1", lambda r: r["bus"] >= 1),
 ("Q_heavy", "heavy traffic with many vehicles", "scene", "total_med >= 13", lambda r: r["tot"] >= 13),
 ("Q_empty", "an almost empty road with hardly any traffic", "scene", "total_med <= 1", lambda r: r["tot"] <= 1),
 ("Q_night_cars", "a night road with several cars", "scene", "night camera and total_med >= 4", lambda r: r["night"] and r["tot"] >= 4),
 ("Q_day_few", "a daytime road with a few cars", "scene", "day, no bus, 3 <= total_med <= 7", lambda r: not r["night"] and r["bus"] == 0 and 3 <= r["tot"] <= 7),
]
Q = [dict(id=i, query=q, category=c, label_source="UA-DETRAC XML ground truth", rule=ru, relevant_clips=[r["c"] for r in R if f(r)]) for i, q, c, ru, f in S]
Q.append(dict(id="Q_near", query="cars close to the camera", category="scene", label_source="manual visual review", rule="manual", relevant_clips=["MVI_20012_005", "MVI_20012_008"]))
json.dump(dict(queries=Q), open("eval/retrieval_gt.json", "w"), indent=2)
print(len(R), "clips read;", [(q["id"], len(q["relevant_clips"])) for q in Q])
