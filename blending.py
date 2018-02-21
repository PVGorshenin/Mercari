pred_mean = 0.2*pred_LGB + 0.4*pred_LR_1 + 0.4*pred_NN

submission = pd.DataFrame({'test_id':test.test_id.astype('int'), 'price':pred_mean})
submission.to_csv('NN_super_mix.csv', index=False)
