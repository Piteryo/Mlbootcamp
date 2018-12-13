from data_processing import process_data

from main import main

from nose.tools import assert_equal, assert_greater
from nose.tools import assert_raises
from nose.tools import raises

class TestA(object):
    @classmethod
    def setup_class(cls):
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.X_submission_df, cls.X_submission = process_data("/home/spolezhaev/train", "/home/spolezhaev/test")
        """This method is run once for each class before any tests are run"""

    @classmethod
    def teardown_class(cls):
        """This method is run once for each class _after_ all tests are run"""

    def setUp(self):
        """Before any test"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    def test_shape(self):
        assert_equal(self.X_train.shape, (104666, 384))
        assert_equal(self.X_test.shape, (34889, 384))
        assert_equal(self.X_submission.shape, (174743, 384))

    def test_score(self):
        assert_greater(main("/home/spolezhaev/train", "/home/spolezhaev/test",
                            "/home/spolezhaev/submissions", False, self.X_train,
                            self.X_test, self.y_train,
                            self.y_test,
                            self.X_submission_df,
                            self.X_submission), 0.50)
    # @raises(KeyError)
    # def test_raise_exc_with_decorator(self):
    #     pass