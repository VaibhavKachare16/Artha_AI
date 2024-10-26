import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from groq import Groq
from typing import List, Dict, Tuple

class EnhancedMutualFundChatbot:
    def __init__(self, csv_file_path):
        """Initialize the chatbot with mutual fund data"""
        self.groq_client = Groq(api_key="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")
        self.model = "llama3-70b-8192"
        
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load and validate mutual fund data
        self.df = self._load_and_validate_data(csv_file_path)
        self.embeddings = None
        self.contexts = None
        self._create_embeddings()  # Create embeddings for the loaded data
        
        # Risk profile definitions
        self.risk_profiles = {
            'conservative': {
                'max_risk_level': 'Low',
                'min_rating': 4,
                'max_beta': 0.8,
                'preferred_categories': ['Debt', 'Hybrid Conservative']
            },
            'moderate': {
                'max_risk_level': 'Moderate',
                'min_rating': 3,
                'max_beta': 1.1,
                'preferred_categories': ['Hybrid', 'Large Cap', 'Index']
            },
            'aggressive': {
                'max_risk_level': 'High',
                'min_rating': 3,
                'max_beta': 1.5,
                'preferred_categories': ['Mid Cap', 'Small Cap', 'Sectoral']
            }
        }

    def _load_and_validate_data(self, csv_file_path) -> pd.DataFrame:
        """Load and validate mutual fund data"""
        df = pd.read_csv(csv_file_path)
        required_columns = [
            'scheme_name', 'category', 'sub_category', 'risk_level',
            'returns_1yr', 'returns_3yr', 'returns_5yr', 'min_sip',
            'min_lumpsum', 'fund_size_cr', 'fund_age_yr', 'expense_ratio',
            'fund_manager', 'rating', 'sharpe', 'alpha', 'beta'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        numeric_cols = ['returns_1yr', 'returns_3yr', 'returns_5yr', 'min_sip',
                       'min_lumpsum', 'fund_size_cr', 'fund_age_yr', 'expense_ratio',
                       'sharpe', 'alpha', 'beta']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def _create_embeddings(self):
        """Create embeddings for mutual fund data"""
        with st.spinner("Creating embeddings for mutual funds..."):
            self.contexts = []
            embeddings_list = []
            
            progress_bar = st.progress(0)
            for idx, row in tqdm(enumerate(self.df.iterrows()), total=len(self.df), desc="Creating embeddings"):
                context = self._create_fund_context(row[1])
                self.contexts.append(context)
                
                tokens = self.tokenizer(context, padding=True, truncation=True, return_tensors='pt')
                with torch.no_grad():
                    outputs = self.embedding_model(**tokens)
                    embeddings_list.append(outputs.last_hidden_state.mean(dim=1))
                
                progress_bar.progress((idx + 1) / len(self.df))
            
            self.embeddings = torch.cat(embeddings_list, dim=0)
            st.success("Embeddings created successfully!")

    def _create_fund_context(self, row: pd.Series) -> str:
        """Create context string for a mutual fund"""
        return (
            f"Fund: {row['scheme_name']} in {row['category']} category. "
            f"Risk level: {row['risk_level']}. "
            f"Returns: {row['returns_1yr']}% (1Y), {row['returns_3yr']}% (3Y), {row['returns_5yr']}% (5Y). "
            f"Investment: Min SIP ₹{row['min_sip']}, Min Lumpsum ₹{row['min_lumpsum']}. "
            f"Fund size: ₹{row['fund_size_cr']} Cr. Age: {row['fund_age_yr']} years. "
            f"Expense ratio: {row['expense_ratio']}%. Manager: {row['fund_manager']}. "
            f"Rating: {row['rating']}. Risk metrics - Sharpe: {row['sharpe']}, "
            f"Alpha: {row['alpha']}, Beta: {row['beta']}."
        )

    def _extract_user_preferences(self, query: str) -> Dict:
        """Extract user preferences from the query using LLM"""
        with st.spinner("Analyzing your preferences..."):
            prompt = (
                f"Based on this investment query: '{query}'\n"
                f"Extract the following information in JSON format:\n"
                "1. Risk profile (conservative/moderate/aggressive)\n"
                "2. Investment horizon (short/medium/long term)\n"
                "3. Investment amount (monthly SIP or lumpsum)\n"
                "4. Preferred categories (if mentioned)\n"
                "5. Special requirements (tax saving, dividend preference, etc.)\n\n"
                "If any information is not provided, mark it as null."
            )

            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a financial query analyzer. Extract only the mentioned parameters in JSON format."},
                          {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )

            try:
                preferences = json.loads(response.choices[0].message.content)
                return preferences
            except json.JSONDecodeError:
                return {
                    "risk_profile": None,
                    "investment_horizon": None,
                    "investment_amount": None,
                    "preferred_categories": None,
                    "special_requirements": None
                }

    def _filter_funds_by_preferences(self, preferences: Dict) -> pd.DataFrame:
        """Filter funds based on user preferences"""
        filtered_df = self.df.copy()

        if preferences['risk_profile']:
            risk_config = self.risk_profiles[preferences['risk_profile'].lower()]
            filtered_df = filtered_df[
                (filtered_df['risk_level'] <= risk_config['max_risk_level']) &
                (filtered_df['rating'] >= risk_config['min_rating']) &
                (filtered_df['beta'] <= risk_config['max_beta'])
            ]

        if preferences['investment_amount']:
            try:
                amount = float(''.join(filter(str.isdigit, preferences['investment_amount'])))
                if 'sip' in preferences['investment_amount'].lower():
                    filtered_df = filtered_df[filtered_df['min_sip'] <= amount]
                else:
                    filtered_df = filtered_df[filtered_df['min_lumpsum'] <= amount]
            except ValueError:
                pass

        if preferences['investment_horizon']:
            if 'short' in preferences['investment_horizon'].lower():
                filtered_df = filtered_df.sort_values('returns_1yr', ascending=False)
            elif 'medium' in preferences['investment_horizon'].lower():
                filtered_df = filtered_df.sort_values('returns_3yr', ascending=False)
            else:
                filtered_df = filtered_df.sort_values('returns_5yr', ascending=False)

        if preferences['preferred_categories']:
            filtered_df = filtered_df[filtered_df['category'].isin(preferences['preferred_categories'])]

        return filtered_df

    def get_personalized_recommendations(self, query: str, top_k: int = 5) -> Tuple[List[pd.Series], Dict]:
        """Get personalized fund recommendations based on user preferences"""
        with st.spinner("Getting personalized recommendations..."):
            preferences = self._extract_user_preferences(query)
            filtered_df = self._filter_funds_by_preferences(preferences)
            
            tokens = self.tokenizer(query, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                query_embedding = self.embedding_model(**tokens).last_hidden_state.mean(dim=1)
            
            filtered_embeddings = torch.tensor(np.array([self.embeddings[i] for i in filtered_df.index]))
            similarities = torch.nn.functional.cosine_similarity(query_embedding, filtered_embeddings)
            
            top_indices = similarities.topk(k=min(top_k, len(filtered_df))).indices.tolist()
            recommendations = [filtered_df.iloc[idx] for idx in top_indices]
            
            return recommendations, preferences

    def generate_personalized_analysis(self, query: str, recommendations: List[pd.Series], preferences: Dict) -> str:
        """Generate personalized analysis using LLM"""
        with st.spinner("Generating personalized analysis..."):
            funds_context = "\n".join([self._create_fund_context(fund) for fund in recommendations])
            
            prompt = (
                f"Based on the user query: '{query}' and the following mutual fund recommendations:\n"
                f"{funds_context}\n"
                "Provide a detailed analysis highlighting the suitability of each fund based on the user's preferences and risk profile."
            )
            
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a financial advisor providing fund analysis."},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()

# Streamlit app
def main():
    st.title("Enhanced Mutual Fund Chatbot")
    
    # Set the path to your CSV file here
    csv_file_path = "mutual_funds_data.csv"
    
    chatbot = EnhancedMutualFundChatbot(csv_file_path)
    
    user_query = st.text_input("Ask about mutual fund recommendations:")
    
    if st.button("Get Recommendations"):
        if user_query:
            recommendations, preferences = chatbot.get_personalized_recommendations(user_query)
            analysis = chatbot.generate_personalized_analysis(user_query, recommendations, preferences)
            
            st.subheader("Recommendations:")
            for fund in recommendations:
                st.write(chatbot._create_fund_context(fund))
            
            st.subheader("Personalized Analysis:")
            st.write(analysis)
        else:
            st.error("Please enter a query.")

if __name__ == "__main__":
    main()
