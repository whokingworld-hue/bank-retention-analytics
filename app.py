import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

st.set_page_config(
    page_title='European Bank',
    page_icon='🏦',
    layout='wide'
)

@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = Path('data/European_Bank.csv')
    if not data_path.exists():
        data_path = Path('European_Bank.csv')

    df = pd.read_csv(data_path)

    # Ensure binary columns are integer type
    for col in ['HasCrCard', 'IsActiveMember', 'Exited']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Engagement profile
    df['EngagementProfile'] = np.where(
        df['IsActiveMember'] == 1,
        'Active',
        np.where(df['Balance'] > 100_000, 'Inactive High Balance', 'Inactive')
    )

    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    geography = st.sidebar.multiselect(
        'Geography',
        options=sorted(df['Geography'].dropna().unique()),
        default=sorted(df['Geography'].dropna().unique())
    )

    gender = st.sidebar.multiselect(
        'Gender',
        options=sorted(df['Gender'].dropna().unique()),
        default=sorted(df['Gender'].dropna().unique())
    )

    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.sidebar.slider(
        'Age range',
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    min_balance, max_balance = int(df['Balance'].min()), int(df['Balance'].max())
    balance_range = st.sidebar.slider(
        'Balance range',
        min_value=min_balance,
        max_value=max_balance,
        value=(min_balance, max_balance),
        step=max(1, int((max_balance - min_balance) / 50))
    )

    min_products, max_products = int(df['NumOfProducts'].min()), int(df['NumOfProducts'].max())
    product_range = st.sidebar.slider(
        'Number of Products',
        min_value=min_products,
        max_value=max_products,
        value=(min_products, max_products)
    )

    filtered = df[
        df['Geography'].isin(geography) &
        df['Gender'].isin(gender) &
        df['Age'].between(age_range[0], age_range[1]) &
        df['Balance'].between(balance_range[0], balance_range[1]) &
        df['NumOfProducts'].between(product_range[0], product_range[1])
    ].copy()

    return filtered


def safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def weighted_churn(df: pd.DataFrame) -> float:
    group = df.groupby('NumOfProducts', as_index=False)['Exited'].agg(['mean', 'count'])
    if group['count'].sum() == 0:
        return 0.0
    return (group['mean'] * group['count']).sum() / group['count'].sum()


def score_tier(score: float) -> str:
    if score <= 30:
        return 'Weak (0-30)'
    if score <= 50:
        return 'Moderate (30-50)'
    if score <= 70:
        return 'Strong (50-70)'
    return 'Very Strong (70-100)'


def make_relationship_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Relationship_Score'] = (
        df['IsActiveMember'] * 30
        + df['HasCrCard'] * 20
        + df['NumOfProducts'].clip(upper=4) * 12.5
        + np.where(df['Tenure'] > 2, 10, 0)
    ).astype(float)
    return df


def format_pct(value: float, digits: int = 1) -> str:
    fmt = f'{{:.{digits}%}}'
    return fmt.format(value)


def main():
    st.title('🏦 European Bank Retention Analytics')
    st.markdown(
        'This dashboard explores customer retention and disengagement patterns for European Bank. '
        'Use the sidebar filters to analyze churn behavior, product depth, high-value disengaged customers, '
        'and retention strength.'
    )

    df = load_data()
    filtered_df = filter_data(df)

    if filtered_df.empty:
        st.warning('No customers match the selected filters. Adjust the filter values to continue.')
        return

    st.markdown('---')

    # KPI cards
    active_df = filtered_df[filtered_df['IsActiveMember'] == 1]
    inactive_df = filtered_df[filtered_df['IsActiveMember'] == 0]
    active_churn_rate = safe_rate(active_df['Exited'].sum(), len(active_df))
    inactive_churn_rate = safe_rate(inactive_df['Exited'].sum(), len(inactive_df))
    engagement_retention_ratio = safe_rate(inactive_churn_rate, active_churn_rate)
    product_depth_index = weighted_churn(filtered_df)
    hvd_df = filtered_df[(filtered_df['Balance'] > 100_000) & (filtered_df['IsActiveMember'] == 0)]
    high_balance_disengagement_rate = safe_rate(hvd_df['Exited'].sum(), len(hvd_df))
    no_card_df = filtered_df[filtered_df['HasCrCard'] == 0]
    card_df = filtered_df[filtered_df['HasCrCard'] == 1]
    credit_card_stickiness = safe_rate(no_card_df['Exited'].sum() / len(no_card_df) if len(no_card_df) else 0,
                                       card_df['Exited'].sum() / len(card_df) if len(card_df) else 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Engagement Retention Ratio', f'{engagement_retention_ratio:.2f}')
    k2.metric('Product Depth Index', f'{product_depth_index:.2%}')
    k3.metric('High-Balance Disengagement Rate', format_pct(high_balance_disengagement_rate, 1))
    k4.metric('Credit Card Stickiness', f'{credit_card_stickiness:.2f}')

    st.markdown('---')
    tabs = st.tabs([
        'Engagement vs Churn',
        'Product Utilization Impact',
        'High-Value Disengaged Customers',
        'Retention Strength Scoring'
    ])

    # Tab 1
    with tabs[0]:
        st.subheader('Engagement vs Churn')
        churn_by_profile = (
            filtered_df.groupby('EngagementProfile', as_index=False)['Exited']
            .mean().sort_values('Exited', ascending=False)
        )
        fig1 = px.bar(
            churn_by_profile,
            x='EngagementProfile',
            y='Exited',
            text=churn_by_profile['Exited'].map('{:.1%}'.format),
            labels={'Exited': 'Churn Rate', 'EngagementProfile': 'Engagement Profile'},
            color='EngagementProfile'
        )
        fig1.update_layout(yaxis_tickformat='.0%', showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.scatter(
                filtered_df,
                x='Balance',
                y='IsActiveMember',
                color=filtered_df['Exited'].map({0: 'Retained', 1: 'Churned'}),
                labels={'IsActiveMember': 'Is Active Member', 'Balance': 'Balance'},
                color_discrete_map={'Churned': 'indianred', 'Retained': 'seagreen'},
                hover_data=['CustomerId', 'Gender', 'Geography']
            )
            fig2.update_yaxes(tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes'])
            fig2.update_layout(plot_bgcolor='white', legend_title_text='Exited')
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.box(
                filtered_df,
                x='IsActiveMember',
                y='Age',
                color=filtered_df['Exited'].map({0: 'Retained', 1: 'Churned'}),
                labels={'IsActiveMember': 'Is Active Member', 'Age': 'Age'},
                category_orders={'IsActiveMember': [0, 1]}
            )
            fig3.update_xaxes(tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes'])
            fig3.update_layout(plot_bgcolor='white', legend_title_text='Exited')
            st.plotly_chart(fig3, use_container_width=True)

    # Tab 2
    with tabs[1]:
        st.subheader('Product Utilization Impact')
        churn_by_products = (
            filtered_df.groupby('NumOfProducts', as_index=False)['Exited']
            .mean()
        )
        fig4 = px.line(
            churn_by_products,
            x='NumOfProducts',
            y='Exited',
            markers=True,
            labels={'Exited': 'Churn Rate', 'NumOfProducts': 'Number of Products'}
        )
        fig4.update_layout(yaxis_tickformat='.0%', plot_bgcolor='white')
        st.plotly_chart(fig4, use_container_width=True)

        depth_labels = []
        depth_values = []
        for label, condition in [
            ('1 product', filtered_df['NumOfProducts'] == 1),
            ('2 products', filtered_df['NumOfProducts'] == 2),
            ('3-4 products', filtered_df['NumOfProducts'].isin([3, 4])),
            ('5+ products', filtered_df['NumOfProducts'] >= 5)
        ]:
            segment = filtered_df[condition]
            depth_labels.append(label)
            depth_values.append(safe_rate(segment['Exited'].sum(), len(segment)))

        depth_df = pd.DataFrame({'Product Depth': depth_labels, 'Churn Rate': depth_values})
        fig5 = px.bar(
            depth_df,
            x='Product Depth',
            y='Churn Rate',
            text=depth_df['Churn Rate'].map('{:.1%}'.format),
            labels={'Churn Rate': 'Churn Rate'}
        )
        fig5.update_layout(yaxis_tickformat='.0%', plot_bgcolor='white', showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            filtered_df['ProductType'] = np.where(filtered_df['NumOfProducts'] == 1, 'Single Product', 'Multi Product')
            single_multi = (
                filtered_df.groupby('ProductType', as_index=False)['Exited']
                .mean()
            )
            fig6 = px.bar(
                single_multi,
                x='ProductType',
                y='Exited',
                text=single_multi['Exited'].map('{:.1%}'.format),
                labels={'Exited': 'Churn Rate', 'ProductType': 'Product Usage'}
            )
            fig6.update_layout(yaxis_tickformat='.0%', plot_bgcolor='white', showlegend=False)
            st.plotly_chart(fig6, use_container_width=True)

        with col2:
            card_split = (
                filtered_df.groupby(['NumOfProducts', 'HasCrCard'], as_index=False)['Exited']
                .mean()
            )
            card_split['HasCrCard'] = card_split['HasCrCard'].map({0: 'No Card', 1: 'Has Card'})
            fig7 = px.line(
                card_split,
                x='NumOfProducts',
                y='Exited',
                color='HasCrCard',
                markers=True,
                labels={'Exited': 'Churn Rate', 'NumOfProducts': 'Number of Products', 'HasCrCard': 'Credit Card'}
            )
            fig7.update_layout(yaxis_tickformat='.0%', plot_bgcolor='white')
            st.plotly_chart(fig7, use_container_width=True)

    # Tab 3
    with tabs[2]:
        st.subheader('High-Value Disengaged Customers')
        high_value = filtered_df[(filtered_df['Balance'] > 100_000) & (filtered_df['IsActiveMember'] == 0)]
        st.markdown(
            'Customers with a high balance and no active membership are especially important for retention outreach.'
        )

        if high_value.empty:
            st.info('No high-value disengaged customers match the current filters.')
        else:
            st.dataframe(
                high_value[
                    ['CustomerId', 'Surname', 'Geography', 'Balance', 'EstimatedSalary', 'NumOfProducts', 'Exited']
                ].sort_values(by='Balance', ascending=False),
                use_container_width=True
            )

            churned_high_value = high_value[high_value['Exited'] == 1]
            if churned_high_value.empty:
                st.info('There are no churned high-value disengaged customers in this filtered selection.')
            else:
                fig8 = px.scatter(
                    churned_high_value,
                    x='Balance',
                    y='EstimatedSalary',
                    color=churned_high_value['IsActiveMember'].map({0: 'Inactive', 1: 'Active'}),
                    labels={'EstimatedSalary': 'Estimated Salary', 'Balance': 'Balance', 'color': 'Is Active Member'},
                    hover_data=['CustomerId', 'Geography', 'NumOfProducts']
                )
                fig8.update_layout(plot_bgcolor='white')
                st.plotly_chart(fig8, use_container_width=True)

            high_balance_share = (
                high_value.groupby('IsActiveMember', as_index=False)['Exited']
                .mean()
            )
            high_balance_share['Status'] = high_balance_share['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})
            fig9 = px.bar(
                high_balance_share,
                x='Status',
                y='Exited',
                text=high_balance_share['Exited'].map('{:.1%}'.format),
                labels={'Exited': 'Churn Rate', 'Status': 'Membership Status'}
            )
            fig9.update_layout(yaxis_tickformat='.0%', plot_bgcolor='white', showlegend=False)
            st.plotly_chart(fig9, use_container_width=True)

    # Tab 4
    with tabs[3]:
        st.subheader('Retention Strength Scoring')
        score_df = make_relationship_score(filtered_df)
        score_df['Score Tier'] = score_df['Relationship_Score'].apply(score_tier)

        fig10 = px.histogram(
            score_df,
            x='Relationship_Score',
            color=score_df['Exited'].map({0: 'Retained', 1: 'Churned'}),
            nbins=20,
            labels={'Relationship_Score': 'Relationship Score', 'color': 'Exited'},
            color_discrete_map={'Retained': 'seagreen', 'Churned': 'indianred'}
        )
        fig10.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig10, use_container_width=True)

        churn_by_tier = (
            score_df.groupby('Score Tier', as_index=False)['Exited']
            .mean()
        )
        tier_order = ['Weak (0-30)', 'Moderate (30-50)', 'Strong (50-70)', 'Very Strong (70-100)']
        churn_by_tier = churn_by_tier.set_index('Score Tier').reindex(tier_order).reset_index()
        churn_by_tier['Exited'] = churn_by_tier['Exited'].fillna(0)
        fig11 = px.bar(
            churn_by_tier,
            x='Score Tier',
            y='Exited',
            text=churn_by_tier['Exited'].map('{:.1%}'.format),
            labels={'Exited': 'Churn Rate'}
        )
        fig11.update_layout(yaxis_tickformat='.0%', plot_bgcolor='white', showlegend=False)
        st.plotly_chart(fig11, use_container_width=True)

        threshold_results = []
        for threshold in range(0, 101, 5):
            subset = score_df[score_df['Relationship_Score'] >= threshold]
            if len(subset) >= 50:
                threshold_results.append({
                    'Threshold': threshold,
                    'Churn Rate': safe_rate(subset['Exited'].sum(), len(subset)),
                    'Customers': len(subset)
                })

        if threshold_results:
            threshold_df = pd.DataFrame(threshold_results)
            fig12 = px.line(
                threshold_df,
                x='Threshold',
                y='Churn Rate',
                markers=True,
                labels={'Churn Rate': 'Churn Rate', 'Threshold': 'Relationship Score Threshold'}
            )
            fig12.update_layout(yaxis_tickformat='.0%', plot_bgcolor='white')
            st.plotly_chart(fig12, use_container_width=True)
        else:
            st.info('At least 50 customers are required to compute threshold churn rates. Adjust filters to increase sample size.')


if __name__ == '__main__':
    main()
