import unittest

import pandas as pd
import numpy as np

from src.temporal_extension import monthly_cohort, valid_metadata
from src.training_resampling import resample_training_indices
from test_uploaded_pilot import row


class TemporalExtensionTests(unittest.TestCase):
    def test_training_resamples_preserve_strata_and_are_reproducible(self):
        labels=np.array([0]*13+[1]*7)
        first=resample_training_indices(labels,100)
        second=resample_training_indices(labels,100)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(np.bincount(labels[first]).tolist(),[13,7])
        self.assertTrue(np.all((first>=0)&(first<len(labels))))
        with self.assertRaisesRegex(ValueError,'Both training classes'):
            resample_training_indices(np.zeros(20),100)

    def test_earlier_removed_activity_excludes_later_authors(self):
        july = row('previous_account','Anxiety','2022-07-04','july',body='[removed]')
        august = row('previous_account','Anxiety','2022-08-04','august')
        other = row('fresh_account','mentalhealth','2022-08-06','fresh')
        metadata,_ = valid_metadata(pd.DataFrame([july,august,other]))
        history={'previous_account':july['created_utc'],'fresh_account':other['created_utc']}
        posts,audit = monthly_cohort(metadata,'mentalhealth','2022-08',{},history,set(),set())
        self.assertEqual(set(posts['author']),{'fresh_account'})
        self.assertEqual(audit['previously_observed_authors_excluded'],1)

    def test_future_affiliation_does_not_change_july_labels(self):
        july=row('july_account','Anxiety','2022-07-02','initial')
        august=row('july_account','mentalhealth','2022-08-02','later')
        july_only,_=valid_metadata(pd.DataFrame([july]))
        combined,_=valid_metadata(pd.DataFrame([july,august]))
        a,_=monthly_cohort(july_only,'mentalhealth','2022-07',{},{},set(),set())
        b,_=monthly_cohort(combined,'mentalhealth','2022-07',{},{},set(),set())
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(b['label'].tolist(),[1])

    def test_reference_reuse_removes_whole_author(self):
        one=row('reusing_account','Anxiety','2022-07-03','one')
        two=row('reusing_account','Anxiety','2022-07-04','two')
        control=row('control_account','mentalhealth','2022-07-04','control')
        metadata,_=valid_metadata(pd.DataFrame([one,two,control]))
        original,_=monthly_cohort(metadata,'mentalhealth','2022-07',{},{},set(),set())
        reused=original.loc[original['author'].eq('reusing_account'),'text'].iloc[0]
        filtered,audit=monthly_cohort(metadata,'mentalhealth','2022-07',{},{},{reused},set())
        self.assertNotIn('reusing_account',set(filtered['author']))
        self.assertEqual(audit['training_validation_exact_reuse_authors_excluded'],1)


if __name__=='__main__':unittest.main()
