    patient_preds = defaultdict(list)
    patient_trues = defaultdict(int)
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1).cpu().tolist()
            trues =  data.y.cpu().tolist()

            sub_ids = data.sub_id

            for i in range(len(sub_ids)):
                sub_id = sub_ids[i]
                patient_preds[sub_id].append(preds[i])
                patient_trues[sub_id] = trues[i]
    
    final_preds = []
    final_trues = []

    for sub_id, preds_list in patient_preds.items():
        count_0 = preds_list.count(0)
        count_1 = preds_list.count(1)

        final_vote = 1 if count_1 > count_0 else 0

        final_preds.append(final_vote)
        final_trues.append(patient_trues[sub_id])

    acc = accuracy_score(final_trues, final_preds)
    f1 = f1_score(final_trues, final_preds, average='macro', zero_division=0)
    recall = recall_score(final_trues, final_preds, average='macro', zero_division=0)
    return acc, f1, recall