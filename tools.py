import torch
import torch.nn as nn

def image_classification_test(loader, base_net, classifier_layer, test_10crop, config, num_iter):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [next(iter_test[j]) for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    features_base = base_net(inputs[j])
                    output = classifier_layer(features_base)
                    outputs.append(nn.Softmax(dim=1)(output))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = classifier_layer(base_net(inputs))

                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float().cpu()), 0)           
    _, predict = torch.max(all_output, 1)
    
    # evaluate model
    class_num = config['network']['params']['class_num']
    shared_class_num = config['network']['params']['shared_class_num']
    class_recall = {}
    class_accuracy = {}
    for i in range(class_num):
        class_index = all_label == i
        predict_index = predict == i
        right_num = torch.sum(torch.squeeze(predict[class_index]).float() == all_label[class_index]).item()
        if right_num == 0:
            class_recall['reccls_'+str(i)] = 0.
            class_accuracy['acccls_'+str(i)] = 0.
            continue
        class_recall['reccls_'+str(i)] = (right_num / float(all_label[class_index].size()[0])) * 100.0
        class_accuracy['acccls_'+str(i)] = (right_num / float(predict[predict_index].size()[0])) * 100.0

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    # metrics for shared class
    share_class_acc = 0.
    share_class_recall = 0.
    for i in range(shared_class_num):
        share_class_acc += class_accuracy['acccls_'+str(i)]
        share_class_recall += class_recall['reccls_'+str(i)]
    share_class_acc /= shared_class_num
    share_class_recall /= shared_class_num
        
    # metrics for target private class
    if shared_class_num != class_num:
        private_class_acc = 0.
        private_class_recall = 0.
        for i in range(shared_class_num, class_num):
            private_class_acc += class_accuracy['acccls_'+str(i)]
            private_class_recall += class_recall['reccls_'+str(i)]
        private_class_acc /= (class_num-shared_class_num)
        private_class_recall /= (class_num-shared_class_num)   
        return accuracy * 100.0, share_class_acc, share_class_recall, private_class_acc, private_class_recall

    else:
        return accuracy * 100.0, share_class_acc, share_class_recall, 0.0, 0.0
