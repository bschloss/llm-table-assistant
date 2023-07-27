import os
import unittest
from llm_table_formatter.utils import load_csv, load_template

class TestLoadCSV(unittest.TestCase):
    def test_table_A(self):
        # Test will either be run from tests folder or root folder
        path = 'testdata/table_A.csv'
        if not os.path.exists(path):
            path = 'tests/testdata/table_A.csv'
        df = load_csv(path)
        column_names = [
            'Date_of_Policy',
            'FullName',
            'Insurance_Plan',
            'Policy_No',
            'Monthly_Premium',
            'Department',
            'JobTitle',
            'Policy_Start',
            'Full_Name',
            'Insurance_Type',
            'Policy_Num',
            'Monthly_Cost'
        ]
        self.assertEqual(column_names, list(df.columns))

    def test_table_B(self):
        # Test will either be run from tests folder or root folder
        path = 'testdata/table_B.csv'
        if not os.path.exists(path):
            path = 'tests/testdata/table_B.csv'
        df = load_csv(path)
        column_names = [
            'PolicyDate',
            'Name',
            'PlanType',
            'Policy_ID',
            'PremiumAmount',
            'Hobby',
            'MaritalStatus',
            'StartDate',
            'Employee_Name',
            'Plan_Name',
            'PolicyID',
            'Cost'
        ]
        self.assertEqual(column_names, list(df.columns))

    def test_table_template(self):
        # Test will either be run from tests folder or root folder
        df = load_template()
        column_names = [
            'Date',
            'EmployeeName',
            'Plan',
            'PolicyNumber',
            'Premium'
        ]
        self.assertEqual(column_names, list(df.columns))


if __name__ == '__main__':
    unittest.main()
