import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.embeds import EmbeddingCache, aggregate_user_embeddings, reduce_dimensions
from src.features import aggregate_user_features, get_feature_names
from src.research_benchmark import audit_posts, paired_comparison, run, chronological_posts, validate_provenance, exact_mcnemar
from src.evaluate import evaluate_model
from src.dataset_prep import prepare_full_dataset
from src.research_stats import hedges_g, benjamini_hochberg, participant_feature_analysis
from scripts.test_external import prepare_external_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def synthetic_posts():
    rows = []
    for split, count in [('train', 12), ('val', 4), ('test', 4)]:
        for label in [0, 1]:
            for user in range(count):
                for position in [1, 2, 3]:
                    content = ('I worry about my future. I feel afraid and sad!' if label else
                               'The team enjoyed a wonderful happy day together.')
                    rows.append({'author': f'{split}_{label}_{user}', 'text': f'{content} {split} unique{user} day{position}',
                                 'label': label, 'split': split, 'timestamp': 1600000000 + position*86400})
    return pd.DataFrame(rows)


class IntegrityTests(unittest.TestCase):
    def test_exact_significance_underflow_retains_log_probability(self):
        result = exact_mcnemar(0, 1200)
        self.assertIsNone(result['pvalue'])
        self.assertTrue(result['pvalue_underflow'])
        self.assertAlmostEqual(result['log10_pvalue'], -1199*np.log10(2.), places=9)
        equal = exact_mcnemar(10,10)
        self.assertEqual(equal['pvalue'],1.)
        self.assertEqual(equal['log10_pvalue'],0.)

    def test_embedding_rows_stay_with_their_authors(self):
        posts = pd.DataFrame({'author': ['b','a','b','a'], 'label':[1,0,1,0],
                              'split':['train']*4, 'posts_seen':[2,2,1,1]})
        x = np.array([[20.,2.],[3.,.3],[10.,1.],[1.,.1]])
        result, y, _ = aggregate_user_embeddings(x, posts)
        np.testing.assert_allclose(result, [[2.,.2],[15.,1.5]])
        np.testing.assert_array_equal(y, [0,1])
        last, _, _ = aggregate_user_embeddings(x, posts, aggregation='last')
        np.testing.assert_allclose(last, [[3.,.3],[20.,2.]])

    def test_conflicting_user_labels_and_leakage_fail(self):
        posts = pd.DataFrame({'author':['same','same'], 'label':[0,1], 'split':['train','test']})
        features = pd.DataFrame({k:[0.,0.] for k in get_feature_names()})
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            aggregate_user_features(features, posts)
        posts['label'] = 0
        with self.assertRaisesRegex(ValueError, 'leakage'):
            aggregate_user_features(features, posts)

    def test_pca_fit_ignores_heldout_distribution(self):
        train = np.array([[0.,0.,0.],[1.,2.,3.],[2.,4.,5.],[3.,5.,7.]])
        first = np.vstack([train, [[4.,6.,8.]]])
        second = np.vstack([train, [[4000.,-6000.,8000.]]])
        reduced1, pca1 = reduce_dimensions(first, 2, training_data=train)
        reduced2, pca2 = reduce_dimensions(second, 2, training_data=train)
        np.testing.assert_allclose(pca1.mean_, train.mean(axis=0))
        np.testing.assert_allclose(pca1.components_, pca2.components_)
        np.testing.assert_allclose(reduced1[:4], reduced2[:4])

    def test_dataset_scaler_and_pca_are_training_only(self):
        posts = synthetic_posts().groupby('author', sort=True).head(1).reset_index(drop=True)
        embeddings = np.random.default_rng(7).normal(size=(len(posts), 5))
        config = {'training': {'embedder':'fixture', 'batch_size':4, 'pca_components':2}}
        with patch('src.dataset_prep.encode_texts', return_value=embeddings):
            first = prepare_full_dataset(posts.copy(), config)
        second_embeds = embeddings.copy()
        second_embeds[posts['split'].ne('train').to_numpy()] += 10000
        with patch('src.dataset_prep.encode_texts', return_value=second_embeds):
            second = prepare_full_dataset(posts.copy(), config)
        np.testing.assert_allclose(first['train_X'], second['train_X'])
        np.testing.assert_allclose(first['scaler'].mean_, second['scaler'].mean_)
        np.testing.assert_allclose(first['reducer'].mean_, second['reducer'].mean_)
        self.assertEqual(len(first['feature_names']), first['train_X'].shape[1])

    def test_one_class_external_sample_does_not_invent_auc(self):
        class Model:
            def predict(self, X): return np.ones(len(X), dtype=int)
            def predict_proba(self, X): return np.tile([.1,.9], (len(X),1))
        result = evaluate_model(Model(), np.zeros((3,2)), np.ones(3), 'one-class fixture')
        self.assertIsNone(result['roc_auc'])
        self.assertEqual(result['confusion_matrix'], [[0,0],[0,3]])

    def test_external_inference_requires_retained_training_transforms(self):
        data=pd.DataFrame({'text':['First sample.','Another sample.'], 'label':[0,1]})
        config={'training':{'embedder':'fixture','batch_size':2,'pca_components':2}}
        with self.assertRaisesRegex(ValueError,'training scaler'):
            prepare_external_data(data.copy(),config)
        training=np.random.default_rng(11).normal(size=(12,5))
        reducer=PCA(n_components=2).fit(training)
        scaler=StandardScaler().fit(np.random.default_rng(12).normal(size=(12,15)))
        saved_components=reducer.components_.copy()
        with patch('scripts.test_external.encode_texts',return_value=training[:2]):
            with self.assertRaisesRegex(ValueError,'TRAINING PCA'):
                prepare_external_data(data.copy(),config,scaler=scaler)
            X,y=prepare_external_data(data.copy(),config,scaler=scaler,reducer=reducer)
        self.assertEqual(X.shape,(2,15))
        np.testing.assert_array_equal(y,[0,1])
        np.testing.assert_allclose(reducer.components_,saved_components)

    def test_cache_checks_content_order_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(directory)
            fingerprint = hashlib.sha256(b'ordered corpus').hexdigest()
            cache.save(np.array([[1.,2.]]), 'model', fingerprint=fingerprint)
            self.assertIsNone(cache.load('model'))
            self.assertIsNone(cache.load('model', fingerprint='same length different texts'))
            np.testing.assert_array_equal(cache.load('model', fingerprint=fingerprint), [[1.,2.]])
            path = cache.get_cache_path('model')
            path.write_bytes(path.read_bytes() + b'corruption')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                cache.load('model', fingerprint=fingerprint)

    def test_duplicate_text_across_splits_fails(self):
        posts = synthetic_posts()
        posts.loc[posts['split'].eq('test').idxmax(), 'text'] = posts.iloc[0]['text'].upper() + ' '
        with self.assertRaisesRegex(ValueError, 'across splits'):
            audit_posts(posts)

    def test_placeholder_authors_are_not_independent_people(self):
        posts = synthetic_posts()
        posts.loc[0, 'author'] = '[deleted]'
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            audit_posts(posts)

    def test_prefix_experiments_require_observed_chronology(self):
        posts=synthetic_posts().drop(columns='timestamp')
        posts['posts_seen']=1
        with self.assertRaisesRegex(ValueError,'observed timestamps'):
            chronological_posts(posts)
        posts=synthetic_posts()
        posts.loc[1,'timestamp']=posts.loc[0,'timestamp']
        with self.assertRaisesRegex(ValueError,'Tied timestamps'):
            chronological_posts(posts)

    def test_real_posts_cannot_use_synthetic_author_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'provenance.json'
            data={'dataset_source':'Original source','dataset_version':'v1', 'permission_basis':'Research permission',
                  'label_definition':'Community membership','ethics_status':'Institutional determination recorded separately',
                  'author_id_origin':'synthetic_grouping','synthetic_data':False}
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'observed_stable'):
                validate_provenance(path)

    def test_paired_statistics_use_the_same_people(self):
        y = np.array([0,0,1,1])
        probability = np.array([.1,.3,.7,.9])
        result = paired_comparison(y, probability, probability, 100)
        self.assertEqual(result['delta']['macro_f1'], 0.)
        self.assertEqual(result['ci95']['macro_f1'], [0.,0.])
        self.assertEqual(result['mcnemar_exact']['pvalue'], 1.)
        weaker = np.array([.8,.8,.7,.9])
        result = paired_comparison(y, probability, weaker, 100)
        self.assertEqual(result['mcnemar_exact']['a_correct_b_wrong'], 2)
        self.assertEqual(result['mcnemar_exact']['a_wrong_b_correct'], 0)
        self.assertLess(result['delta']['macro_f1'], 0)

    def test_clinical_effects_require_real_independent_measurements(self):
        self.assertGreater(hedges_g([3,4,5],[0,1,2]), 0)
        self.assertEqual(hedges_g([0,1,2],[0,1,2]), 0.)
        np.testing.assert_allclose(benjamini_hochberg([.01,.04,.03]), [.03,.04,.04])
        data=pd.DataFrame({'participant_id':['a','b','c','d'], 'label':[0,0,1,1], 'sentiment':[.1,.2,.5,.6]})
        result=participant_feature_analysis(data,['sentiment'])
        self.assertGreater(result.iloc[0]['hedges_g_positive_minus_control'],0)
        duplicate=pd.concat([data,data.iloc[:1]],ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'one row'):
            participant_feature_analysis(duplicate,['sentiment'])

    def test_end_to_end_is_explicitly_synthetic_and_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_posts().to_csv(root/'input.csv', index=False)
            provenance = {'dataset_source':'Generated software fixture', 'dataset_version':'1',
                          'permission_basis':'Generated synthetic text', 'label_definition':'Synthetic class assignment',
                          'ethics_status':'Software fixture; no human participants',
                          'author_id_origin':'generated_fixture', 'synthetic_data': True}
            (root/'provenance.json').write_text(json.dumps(provenance))
            report = run(root/'input.csv', root/'provenance.json', root/'output', [3], 100)
            self.assertEqual(report['status'], 'synthetic_software_validation')
            self.assertEqual(set(report['experiments']), {'full', '3'})
            full = pd.read_csv(root/'output/full_predictions.csv')
            prefix = pd.read_csv(root/'output/3_predictions.csv')
            self.assertEqual(full['author_id'].tolist(), prefix['author_id'].tolist())
            self.assertEqual(report['experiments']['full']['metrics']['linguistic_13']['n_users'], 8)
            pipeline = joblib.load(root/'output/full_linguistic_13.joblib')
            self.assertIn('standardscaler', pipeline.named_steps)
            self.assertTrue((root/'output/report.json').exists())
            self.assertNotIn('text', full.columns)
            with self.assertRaises(FileExistsError):
                run(root/'input.csv', root/'provenance.json', root/'output', [], 100)


if __name__ == '__main__':
    unittest.main()
