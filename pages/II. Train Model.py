import numpy as np                
import pandas as pd               
from sklearn.model_selection import train_test_split
import streamlit as st             
import random
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
#############################################

# Models: Linear, Polynomial, Ridge, Lasso Regression + Naive Bayes (IGN binary classification)

SESSION_DATA_KEY = "game_df"
LEGACY_SESSION_DATA_KEY = "house_df"

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Game Score Prediction and Critics Analysis")

#############################################

st.title('Train Model')

#############################################


def split_dataset(X, y, number,random_state=45):
    """
    This function splits the dataset into 4 parts–- feature 
    and target sets for training and validation.

    Input: 
        - X: training features
        - y: training targets
        - number: the ratio of test samples
    Output: 
        - X_train: training features
        - X_val: test/validation features
        - y_train: training targets
        - y_val: test/validation targets
    """
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    
    # Add code here
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=number/100, random_state=random_state)

    return X_train, X_val, y_train, y_val


class LinearRegression(object) : 
    def __init__(self, learning_rate=0.001, num_iterations=500): 
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.cost_history=[]

    
    def predict(self, X): 
        '''
        Make a housing price prediction using model weights and input features X
        
        Input
        - X: matrix of column-wise features
        Output
        - prediction: prediction of house price
        '''
        prediction=None
        # Add code here
        X_norm = self.normalize(X)

        bias = self.W[0]
        weights = self.W[1:]

        linear_part = np.dot(X_norm, weights)

        prediction = linear_part + bias

        prediction = prediction.reshape(-1, 1)

        return prediction

    def update_weights(self):     
        '''
        Update weights of regression model by computing the 
        derivative of the RSS cost function with respect to weights
        
        Input: None
        Output: None
        ''' 
        # Add code here
        self.W = self.W.reshape(-1, 1)
        X_norm = self.normalize(self.X)

        m = X_norm.shape[0]
        Y = self.Y.reshape(-1, 1)
        Y_pred = self.predict(self.X).reshape(-1, 1)
        error = Y_pred - Y

        dW0 = (2 / m) * np.sum(error)
        dW_rest = (2 / m) * (X_norm.T @ error)
        gradient = np.vstack((dW0.reshape(1, 1), dW_rest))


        self.W = self.W - self.learning_rate * gradient


        cost = np.sum((Y - Y_pred) ** 2)
        self.cost_history.append(cost)

        return self
    
    def fit(self, X, Y): 
        '''
        Use gradient descent to update the weights for self.num_iterations
        
        Input
            - X: Input features X
            - Y: True values of housing prices
        Output: None
        '''
        self.X = X 
        self.Y = Y
        # Add code here
        self.num_features = self.X.shape[1]
        self.W = np.zeros((self.num_features + 1, 1)) 
        
        self.num_examples = self.X.shape[0]
        
        for i in range(self.num_iterations):
            self.update_weights()

        return self
    
    # Helper function
    def normalize(self, X):
        '''
        Standardize features X by column

        Input: X is input features (column-wise)
        Output: Standardized features by column
        '''
        X_normalized=X
        # Add code here
        X = X.astype(np.float64)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
    
        return X_normalized
    
    def get_weights(self, model_name, features):
        '''
        This function prints the weights of a regression model in out_dict using the model name as a key.

        Input:
            - model_name: (string) name of model
            - features: input features
        Output:
            - out_dict: a dictionary contains the coefficients of the selected models, with the following keys:
            - 'Multiple Linear Regression'
            - 'Polynomial Regression'
            - 'Ridge Regression'
            - 'Lasso Regression'
        '''
        out_dict = {'Multiple Linear Regression': [],
                'Polynomial Regression': [],
                'Ridge Regression': [],
                'Lasso Regression': []}
        # Add code here
        out_dict[model_name] = self.W

        st.write(model_name)
        st.write("Weights (bias + feature weights):")
        st.write(self.W)
        return out_dict

# Multivariate Polynomial Regression
class PolynomialRegression(LinearRegression):
    def __init__(self, degree, learning_rate, num_iterations):
        self.degree = degree

        # invoking the __init__ of the parent class
        LinearRegression.__init__(self, learning_rate, num_iterations)

    # Helper function
    def transform(self, X):
        '''
        Converts a matrix of features for polynomial  h( x ) = w0 * x^0 + w1 * x^1 + w2 * x^2 + ........+ wn * x^n

        Input
            - X:
        Output
            - X_transform:
        '''
        X_transform=[]
        # Add code here
        num_features = X.shape[1]
        
        from itertools import combinations_with_replacement
        
        X_transform = []
        
        for sample in X:
            poly_features = []
            for d in range(self.degree + 1):
                for combo in combinations_with_replacement(range(num_features), d):
                    term = 1.0
                    for idx in combo:
                        term *= sample[idx]
                    poly_features.append(term)
            X_transform.append(poly_features)
        
        X_transform = np.array(X_transform)

        # from sklearn.preprocessing import PolynomialFeatures

        # poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        # X_transform = poly.fit_transform(X)

        return X_transform
    
    def fit(self, X, Y):
        '''
        Use gradient descent to update the weights for self.num_iterations

        Input:
            - X: Input features X
            - Y: True values of housing prices
        Output: None
        '''
        self.X = X
        self.Y = Y
        # Add code here
        X_poly = self.transform(X)

        X_norm = np.array(X_poly, dtype=float)
        mean = np.mean(X_norm[:, 1:], axis=0)
        std = np.std(X_norm[:, 1:], axis=0)
        X_norm[:, 1:] = (X_norm[:, 1:] - mean) / std

        self.X = X_norm
        self.Y = Y

        self.num_examples = self.X.shape[0]
        self.num_features = self.X.shape[1]

        self.W = np.zeros((self.num_features, 1))

        self.cost_history = []

        for i in range(self.num_iterations):
            Y_pred = np.dot(self.X, self.W)

            error = Y_pred - self.Y
            grad = (2 / self.num_examples) * np.dot(self.X.T, error)

            self.W = self.W - self.learning_rate * grad

            cost = np.sum((self.Y - Y_pred) ** 2)
            self.cost_history.append(cost)

        return self
    
    def predict(self, X):
        '''
        Make a prediction using coefficients self.W and input features X
        
        Input
        - X: matrix of column-wise features
        Output
        - prediction: prediction of house price
        '''
        prediction=None
        # Add code here
        if X.shape[1] != self.W.shape[0]:
            X_poly = self.transform(X)

            X_poly = np.array(X_poly, dtype=float)
            mean = np.mean(X_poly[:, 1:], axis=0)
            std = np.std(X_poly[:, 1:], axis=0)
            X_poly[:, 1:] = (X_poly[:, 1:] - mean) / std

            X_use = X_poly
        else:
            X_use = X

        prediction = np.dot(X_use, self.W)

        return prediction

# Backward compatibility alias for typo in original class name
PolynomailRegression = PolynomialRegression

# Ridge Regression
class RidgeRegression(LinearRegression): 
    def __init__(self, learning_rate, num_iterations, l2_penalty): 
        self.l2_penalty = l2_penalty 

        # invoking the __init__ of the parent class
        LinearRegression.__init__(self, learning_rate, num_iterations)

    def update_weights(self):      
        '''
        Update weights of regression model by computing the 
        derivative of the RSS + l2_penalty*w cost function with respect to weights

        Input: None
        Output: None
        '''
        # Add code here
        X = self.X.astype(np.float32)
        Y = self.Y.astype(np.float32)

        m = X.shape[0]
        X_norm = self.normalize(X).astype(np.float32)

        # prediction
        W = self.W.astype(np.float32).reshape(-1, 1)
        bias = W[0]
        weights = W[1:]
        Y_pred = (bias + np.dot(X_norm, weights)).reshape(-1, 1)

        # gradients
        error = Y_pred - Y

        dW0 = (2 / m) * np.sum(error)
        dW_rest = (2 / m) * np.dot(X_norm.T, error)

        dW0 = np.array([[dW0]], dtype=np.float32)
        grad_rss = np.vstack((dW0, dW_rest)).astype(np.float32)

        grad = grad_rss - self.l2_penalty * W

        # update (keep as float32 then assign back)
        self.W = (W - self.learning_rate * grad).astype(np.float32)

        cost = np.sum((Y - Y_pred) ** 2) + self.l2_penalty * np.sum(self.W ** 2)
        self.cost_history.append(cost)

        return self

# Lasso Regression 
class LassoRegression(LinearRegression): 
    def __init__(self, learning_rate, num_iterations, l1_penalty): 
        self.l1_penalty = l1_penalty 

        # invoking the __init__ of the parent class
        LinearRegression.__init__(self, learning_rate, num_iterations)

    def update_weights(self):      
        '''
        Compute the derivative and update model weights 

        Input: None
        Output: None
        '''
        # Add code here
         
        self.W = self.W.reshape(-1, 1)
        X_norm = self.normalize(self.X)
        
        m = X_norm.shape[0]
        Y = self.Y.reshape(-1, 1)
        Y_pred = self.predict(self.X).reshape(-1, 1)
        error = Y_pred - Y
        
        # Step 2: Compute gradient with L1 regularization
        dW0 = (2 / m) * np.sum(error)
        # Add L1 penalty term (λ * sign(w)) to feature weights only
        dW_rest = (2 / m) * (X_norm.T @ error) + self.l1_penalty * np.sign(self.W[1:])
        gradient = np.vstack((dW0.reshape(1, 1), dW_rest))
        
        # Step 3: Update weights
        self.W = self.W - self.learning_rate * gradient
        
        # Store cost with L1 penalty
        cost = np.sum((Y - Y_pred) ** 2) + self.l1_penalty * np.sum(np.abs(self.W[1:]))
        self.cost_history.append(cost)
        
        return self

# Naive Bayes Classifier
class NaiveBayes(object):
    def __init__(self, classes, alpha=1):
        self.model_name = 'Naive Bayes'
        self.classes = classes
        if not isinstance(self.classes, np.ndarray):
            self.classes = np.array(self.classes)
        self.num_classes = len(self.classes)
        mapping = {i: k for i, k in enumerate(self.classes)}
        self.idx_to_class = np.vectorize(mapping.get)
        self.likelihood_history = []
        self.alpha = alpha
        self.W = []
        self.W_prior = []

    def predict_logprob(self, X):
        """
        Computes the log probability of each class given input features.
        Input:  X – input features
        Output: y_pred – log-probability matrix (n_samples × n_classes)
        """
        y_pred = None
        try:
            X = np.array(X)
            y_pred = X @ np.log(self.W).T + np.log(self.W_prior)
        except ValueError as err:
            st.write({str(err)})
        return y_pred

    def predict_probability(self, X):
        """
        Probabilistic estimate P(y = High | x).
        Input:  X – input features
        Output: y_pred – probability of High class (0–1)
        """
        y_pred = None
        try:
            X = X - X.min()
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            pos_class_idx = np.where(self.classes == 1)[0][0]
            y_pred = self.predict_logprob(X)
            probs = np.exp(np.array(y_pred))
            probs = np.exp(probs) / np.sum(np.exp(probs), axis=1)[:, None]
            y_pred = probs[:, pos_class_idx]
        except ValueError as err:
            st.write({str(err)})
        return y_pred

    def predict(self, X):
        """
        Predicts binary class label (0=Low, 1=High) for each sample.
        Input:  X – input features
        Output: y_pred – array of predicted class labels
        """
        y_pred = None
        try:
            X = X - X.min()
            y_pred = self.predict_logprob(X)
            y_pred = np.argmax(y_pred, axis=1)
            mapping = {i: k for i, k in enumerate(self.classes)}
            idx_to_class = np.vectorize(mapping.get)
            y_pred = idx_to_class(y_pred)
        except ValueError as err:
            st.write({str(err)})
        return y_pred

    def fit(self, X, Y):
        """
        Closed-form Naive Bayes fit with Laplace smoothing.
        Input:  X – features, Y – binary labels (0/1)
        Output: self (trained model)
        """
        try:
            X = X - X.min()  # shift to non-negative for Naive Bayes
            num_examples, num_features = X.shape
            self.W = np.zeros((self.num_classes, num_features))
            self.W_prior = np.zeros(self.num_classes)
            for ind, class_k in enumerate(self.classes):
                X_class_k = X[Y == class_k]
                self.W[ind] = (np.sum(X_class_k, axis=0) + self.alpha)
                self.W[ind] /= (np.sum(X_class_k) + (self.alpha * num_features))
                self.W_prior[ind] = X_class_k.shape[0] / num_examples
            self.W = np.clip(self.W, 1e-10, 1.0)
            self.W_prior = np.clip(self.W_prior, 1e-10, 1.0)
            log_likelihood = np.log(self.predict_probability(X)).mean()
            self.likelihood_history.append(log_likelihood)
        except ValueError as err:
            st.write({str(err)})
        return self

    def get_weights(self):
        """Prints and returns trained model weights."""
        weights = None
        try:
            if len(self.W):
                st.write('-------------------------')
                st.write('Model Coefficients for ' + self.model_name)
                num_positive_weights = np.sum(self.W >= 0) + np.sum(self.W_prior >= 0)
                num_negative_weights = np.sum(self.W < 0) + np.sum(self.W_prior < 0)
                st.write('* Number of positive weights: {}'.format(num_positive_weights))
                st.write('* Number of negative weights: {}'.format(num_negative_weights))
                weights = [self.W, self.W_prior]
            else:
                st.write('There are no model weights to print. Train a model first.')
        except ValueError as err:
            st.write(str(err))
        return weights


# Helper functions
def load_dataset(filepath):
    '''
    This function uses the filepath (string) a .csv file locally on a computer 
    to import a dataset with pandas read_csv() function. Then, store the 
    dataset in session_state.

    Input: data is the filename or path to file (string)
    Output: pandas dataframe df
    '''
    try:
        data = pd.read_csv(filepath)
        st.session_state[SESSION_DATA_KEY] = data
        st.session_state[LEGACY_SESSION_DATA_KEY] = data
    except ValueError as err:
            st.write({str(err)})
    return data

random.seed(10)
###################### FETCH DATASET #######################
df = None
filepath = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
if(filepath):
    df = load_dataset(filepath)

if(SESSION_DATA_KEY in st.session_state):
    df = st.session_state[SESSION_DATA_KEY]
elif(LEGACY_SESSION_DATA_KEY in st.session_state):
    df = st.session_state[LEGACY_SESSION_DATA_KEY]

###################### DRIVER CODE #######################

if df is not None:
    # Display dataframe as table
    st.dataframe(df.describe())

    # Select variable to predict
    feature_predict_select = st.selectbox(
        label='Select variable to predict',
        options=list(df.select_dtypes(include='number').columns),
        key='feature_selectbox',
        index=8
    )

    st.session_state['target'] = feature_predict_select

    # Select input features
    feature_input_select = st.multiselect(
        label='Select features for regression input',
        options=[f for f in list(df.select_dtypes(
            include='number').columns) if f != feature_predict_select],
        key='feature_multiselect'
    )

    st.session_state['feature'] = feature_input_select

    st.write('You selected input {} and output {}'.format(
        feature_input_select, feature_predict_select))

    df = df.dropna()
    X = df.loc[:, df.columns.isin(feature_input_select)]
    Y = df.loc[:, df.columns.isin([feature_predict_select])]

    # Split train/test
    st.markdown('## Split dataset into Train/Test sets')
    st.markdown(
        '### Enter the percentage of test data to use for training the model')
    split_number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    # Compute the percentage of test and training data
    X_train_df, X_val_df, y_train_df, y_val_df = split_dataset(X, Y, split_number)
    if(len(X_train_df)!=0):
        st.session_state['X_train_df'] = X_train_df
        st.session_state['X_val_df'] = X_val_df
        st.session_state['y_train_df'] = y_train_df
        st.session_state['y_val_df'] = y_val_df

        # Convert to numpy arrays
        X = np.asarray(X.values.tolist()) 
        Y = np.asarray(Y.values.tolist()) 
        X_train, X_val, y_train, y_val = split_dataset(X, Y, split_number)
        train_percentage = (len(X_train) / (len(X_train)+len(y_val)))*100
        test_percentage = (len(X_val)) / (len(X_train)+len(y_val))*100

        st.markdown('Training dataset ({1:.2f}%): {0:.2f}'.format(len(X_train),train_percentage))
        st.markdown('Test dataset ({1:.2f}%): {0:.2f}'.format(len(X_val),test_percentage))
        st.markdown('Total number of observations: {0:.2f}'.format(len(X_train)+len(y_val)))
        train_percentage = (len(X_train)+len(y_train) /
                            (len(X_train)+len(X_val)+len(y_train)+len(y_val)))*100
        test_percentage = ((len(X_val)+len(y_val)) /
                            (len(X_train)+len(X_val)+len(y_train)+len(y_val)))*100

    regression_methods_options = ['Multiple Linear Regression',
                                  'Polynomial Regression',
                                  'Ridge Regression',
                                  'Lasso Regression',
                                  'Naive Bayes']
    # Collect ML Models of interests
    regression_model_select = st.multiselect(
        label='Select model for prediction / classification',
        options=regression_methods_options,
    )
    st.write('You selected the follow models: {}'.format(
        regression_model_select))

    # Multiple Linear Regression
    if (regression_methods_options[0] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[0])

        # Add parameter options to each regression method
        learning_rate_input = st.text_input(
            label='Input learning rate 👇',
            value='0.01',
            key='mr_alphas_textinput'
        )
        st.write('You select the following alpha value(s): {}'.format(learning_rate_input))

        num_iterations_input = st.text_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            value='100',
            key='mr_iter_textinput'
        )
        st.write('You select the following number of iteration value(s): {}'.format(num_iterations_input))

        multiple_reg_params = {
            'num_iterations': [float(val) for val in num_iterations_input.split(',')],
            'alpha': [float(val) for val in learning_rate_input.split(',')]
        }

        if st.button('Train Multiple Linear Regression Model'):
            # Handle errors
            try:
                multi_reg_model = LinearRegression(learning_rate=multiple_reg_params['alpha'][0], 
                                                   num_iterations=int(multiple_reg_params['num_iterations'][0]))
                multi_reg_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[0]] = multi_reg_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[0] not in st.session_state:
            st.write('Multiple Linear Regression Model is untrained')
        else:
            st.write('Multiple Linear Regression Model trained')

    # Polynomial Regression
    if (regression_methods_options[1] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[1])

        poly_degree = st.number_input(
            label='Enter the degree of polynomial',
            min_value=0,
            max_value=1000,
            value=3,
            step=1,
            key='poly_degree_numberinput'
        )
        st.write('You set the polynomial degree to: {}'.format(poly_degree))

        poly_num_iterations_input = st.number_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            min_value=1,
            max_value=10000,
            value=50,
            step=1,
            key='poly_num_iter'
        )
        st.write('You set the polynomial degree to: {}'.format(poly_num_iterations_input))

        poly_input=[0.001]
        poly_learning_rate_input = st.text_input(
            label='Input learning rate 👇',
            value='0.0001',
            key='poly_alphas_textinput'
        )
        st.write('You select the following alpha value(s): {}'.format(poly_learning_rate_input))

        poly_reg_params = {
            'num_iterations': poly_num_iterations_input,
            'alphas': [float(val) for val in poly_learning_rate_input.split(',')],
            'degree' : poly_degree
        }

        if st.button('Train Polynomial Regression Model'):
            # Handle errors
            try:
                poly_reg_model = PolynomialRegression(poly_reg_params['degree'], 
                                                      poly_reg_params['alphas'][0], 
                                                      poly_reg_params['num_iterations'])
                poly_reg_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[1]] = poly_reg_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[1] not in st.session_state:
            st.write('Polynomial Regression Model is untrained')
        else:
            st.write('Polynomial Regression Model trained')

    # Ridge Regression
    if (regression_methods_options[2] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[2])

        # Add parameter options to each regression method
        ridge_alphas = st.text_input(
            label='Input learning rate 👇',
            value='0.01',
            key='ridge_lr_textinput'
        )
        st.write('You select the following learning rate: {}'.format(ridge_alphas))

        ridge_num_iterations_input = st.text_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            value='100',
            key='ridge_num_iter'
        )
        st.write('You set the number of iterations to: {}'.format(ridge_num_iterations_input))

        ridge_l2_penalty_input = st.text_input(
            label='Enter the l2 penalty (0-1)👇',
            value='1',
            key='ridge_l2_penalty_textinput'
        )
        st.write('You select the following l2 penalty value: {}'.format(ridge_l2_penalty_input))

        ridge_params = {
            'num_iterations': [int(val) for val in ridge_num_iterations_input.split(',')],
            'learning_rate': [float(val) for val in ridge_alphas.split(',')],
            'l2_penalty':[float(val) for val in ridge_l2_penalty_input.split(',')]
        }
        if st.button('Train Ridge Regression Model'):
            # Train ridge on all feature --> feature selection
            # Handle Errors
            try:
                ridge_model = RidgeRegression(learning_rate=ridge_params['learning_rate'][0],
                                           num_iterations=ridge_params['num_iterations'][0],
                                           l2_penalty=ridge_params['l2_penalty'][0])
                ridge_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[2]] = ridge_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[2] not in st.session_state:
            st.write('Ridge Model is untrained')
        else:
            st.write('Ridge Model trained')

    # Lasso Regression
    if (regression_methods_options[3] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[3])

        # Add parameter options to each regression method
        lasso_alphas = st.text_input(
            label='Input learning rate 👇',
            value='0.0001',
            key='lasso_lr_textinput'
        )
        st.write('You select the following learning rate: {}'.format(lasso_alphas))

        lasso_num_iterations_input = st.text_input(
            label='Enter the number of iterations to run Gradient Descent (seperate with commas)👇',
            value='100',
            key='lasso_num_iter'
        )
        st.write('You set the number of iterations to: {}'.format(lasso_num_iterations_input))

        lasso_l1_penalty_input = st.text_input(
            label='Enter the l1 penalty (0-1)👇',
            value='0.5',
            key='lasso_l1_penalty_textinput'
        )
        st.write('You select the following l1 penalty value: {}'.format(lasso_l1_penalty_input))

        lasso_params = {
            'num_iterations': [int(val) for val in lasso_num_iterations_input.split(',')],
            'learning_rate': [float(val) for val in lasso_alphas.split(',')],
            'l1_penalty':[float(val) for val in lasso_l1_penalty_input.split(',')]
        }
        if st.button('Train Lasso Regression Model'):
            # Train lasso on all feature --> feature selection
            # Handle Errors
            try:
                lasso_model = LassoRegression(learning_rate=lasso_params['learning_rate'][0],
                                                num_iterations=lasso_params['num_iterations'][0],
                                                l1_penalty=lasso_params['l1_penalty'][0])
                lasso_model.fit(X_train, y_train)
                st.session_state[regression_methods_options[3]] = lasso_model
            except ValueError as err:
                st.write({str(err)})

        if regression_methods_options[3] not in st.session_state:
            st.write('Lasso Model is untrained')
        else:
            st.write('Lasso Model trained')

    # Store models
    trained_models={}
    for model_name in regression_model_select:
        if(model_name in st.session_state):
            trained_models[model_name] = st.session_state[model_name]

    # Inspect Regression coefficients
    st.markdown('## Inspect model coefficients')

    # Select multiple models to inspect
    inspect_models = st.multiselect(
        label='Select model',
        options=regression_model_select,
        key='inspect_multiselect'
    )
    st.write('You selected the {} models'.format(inspect_models))
    
    models = {}
    weights_dict = {}
    if(inspect_models):
        for model_name in inspect_models:
            model = trained_models.get(model_name)
            if model is None:
                st.write('{} is untrained'.format(model_name))
            elif model_name == 'Naive Bayes':
                weights_dict = model.get_weights()
            else:
                weights_dict = model.get_weights(model_name, feature_input_select)

    # Inspect model cost
    st.markdown('## Inspect model cost')

    # Select multiple models to inspect
    inspect_model_cost = st.selectbox(
        label='Select model',
        options=regression_model_select,
        key='inspect_cost_multiselect'
    )

    st.write('You selected the {} model'.format(inspect_model_cost))

    if(inspect_model_cost):
        try:
            if trained_models.get(inspect_model_cost) is None:
                st.write('{} is untrained'.format(inspect_model_cost))
                raise ValueError('model not trained')
            fig = make_subplots(rows=1, cols=1,
                shared_xaxes=True, vertical_spacing=0.1)
            cost_history=trained_models[inspect_model_cost].cost_history

            x_range = st.slider("Select x range:",
                                    value=(0, len(cost_history)))
            st.write("You selected : %d - %d"%(x_range[0],x_range[1]))
            cost_history_tmp = cost_history[x_range[0]:x_range[1]]
            
            fig.add_trace(go.Scatter(x=np.arange(x_range[0],x_range[1],1),
                        y=cost_history_tmp, mode='markers', name=inspect_model_cost), row=1, col=1)

            fig.update_xaxes(title_text="Training Iterations")
            fig.update_yaxes(title_text='Cost', row=1, col=1)
            fig.update_layout(title=inspect_model_cost)
            st.plotly_chart(fig)
        except Exception as e:
            print(e)

    # Naive Bayes Classification
    if (regression_methods_options[4] in regression_model_select):
        st.markdown('#### ' + regression_methods_options[4])
        st.markdown(
            'Binary classification: **High (1)** if score ≥ threshold, **Low (0)** otherwise.'
        )

        nb_score_options = [c for c in df.select_dtypes(include='number').columns
                            if c in ('IGN', 'Avg_Reviews', 'Metacritic', 'GameSpot', 'Destructoid')]
        if not nb_score_options:
            nb_score_options = list(df.select_dtypes(include='number').columns)

        nb_target_col = st.selectbox(
            'Score column to binarize',
            options=nb_score_options,
            index=nb_score_options.index('IGN') if 'IGN' in nb_score_options else 0,
            key='nb_target_col'
        )

        nb_threshold = st.slider(
            'Threshold (≥ threshold → High=1, else Low=0)',
            min_value=0.0, max_value=10.0, value=7.5, step=0.5,
            key='nb_threshold'
        )
        st.write('Games with {} ≥ {:.1f} → **High (1)**, otherwise → **Low (0)**'.format(
            nb_target_col, nb_threshold))

        nb_alpha_input = st.number_input(
            'Laplace smoothing alpha', min_value=0.0, value=1.0, step=0.1, key='nb_alpha'
        )

        nb_feature_select = st.multiselect(
            'Select features for Naive Bayes input',
            options=[f for f in list(df.select_dtypes(include='number').columns)
                     if f != nb_target_col],
            key='nb_feature_multiselect'
        )

        if st.button('Train Naive Bayes Model'):
            if nb_feature_select:
                try:
                    nb_cols = nb_feature_select + [nb_target_col]
                    df_nb = df.dropna(subset=nb_cols)
                    X_nb = df_nb[nb_feature_select].values.astype(float)
                    y_nb = (df_nb[nb_target_col] >= nb_threshold).astype(int).values

                    high_count = int(y_nb.sum())
                    low_count = len(y_nb) - high_count
                    st.write('Class distribution — High: {}, Low: {}, Total: {}'.format(
                        high_count, low_count, len(y_nb)))

                    X_nb_train, X_nb_val, y_nb_train, y_nb_val = split_dataset(
                        X_nb, y_nb, split_number)

                    nb_model = NaiveBayes(classes=np.array([0, 1]), alpha=nb_alpha_input)
                    nb_model.fit(X_nb_train, y_nb_train)
                    st.session_state[regression_methods_options[4]] = nb_model
                    st.success('Naive Bayes Model trained!')
                except Exception as e:
                    st.write(str(e))
            else:
                st.warning('Please select at least one feature.')

        if regression_methods_options[4] not in st.session_state:
            st.write('Naive Bayes Model is untrained')
        else:
            st.write('Naive Bayes Model trained')
            st.session_state[regression_methods_options[4]].get_weights()

    st.write('Continue to Test Model')
