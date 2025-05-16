import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class POSAnalysisModel:
    def __init__(self, transactions_path, customers_path, output_dir='output'):
        """
        Initialize the POS Analysis Model with transaction and customer data
        
        Parameters:
        -----------
        transactions_path : str
            Path to the CSV file containing transaction data
        customers_path : str
            Path to the CSV file containing customer data
        output_dir : str
            Directory to save output files
        """
        self.transactions_path = transactions_path
        self.customers_path = customers_path
        self.output_dir = output_dir
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
        # Initialize data frames
        self.transactions_df = None
        self.customers_df = None
        self.merged_df = None
        self.monthly_customer_values = None
        self.customer_segments = None
        
        # Set style for visualizations
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def load_data(self):
        """Load transaction and customer data from CSV files"""
        try:
            self.transactions_df = pd.read_csv(self.transactions_path)
            self.customers_df = pd.read_csv(self.customers_path)
            
            # Convert transaction_date to datetime
            self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'])
            
            # Add month column for monthly analysis
            self.transactions_df['month'] = self.transactions_df['transaction_date'].dt.to_period('M').astype(str)
            
            print(f"Loaded {len(self.transactions_df)} transactions and {len(self.customers_df)} customer records")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def merge_data(self):
        """Merge transaction and customer data"""
        try:
            # Merge transactions with customer data
            self.merged_df = pd.merge(
                self.transactions_df,
                self.customers_df,
                on=['customer_id'],
                how='left'
            )
            print(f"Merged data contains {len(self.merged_df)} records")
            return True
            
        except Exception as e:
            print(f"Error merging data: {e}")
            return False
    
    def calculate_pos_adoption(self):
        """Calculate POS terminal adoption rate"""
        # Customers with transactions
        customers_with_pos = self.transactions_df['customer_id'].nunique()
        
        # Total customers
        total_customers = self.customers_df['customer_id'].nunique()
        
        # Calculate adoption rate
        adoption_rate = customers_with_pos / total_customers
        
        return {
            'customers_with_pos': customers_with_pos,
            'total_customers': total_customers,
            'adoption_rate': adoption_rate,
            'adoption_percentage': adoption_rate * 100
        }
    
    def calculate_pos_adoption_by_location(self):
        """Calculate POS adoption rate by location"""
        # Get customers with transactions
        customers_with_pos = set(self.transactions_df['customer_id'].unique())
        
        # Create a new column in customers_df indicating if customer has POS
        self.customers_df['has_pos'] = self.customers_df['customer_id'].isin(customers_with_pos)
        
        # Group by location and calculate adoption rate
        adoption_by_location = self.customers_df.groupby('location').agg(
            customers_with_pos=('has_pos', 'sum'),
            total_customers=('customer_id', 'nunique')
        ).reset_index()
        
        # Calculate adoption rate
        adoption_by_location['adoption_rate'] = adoption_by_location['customers_with_pos'] / adoption_by_location['total_customers']
        adoption_by_location['adoption_percentage'] = adoption_by_location['adoption_rate'] * 100
        
        return adoption_by_location
    
    def calculate_pos_adoption_by_division(self, divisions):
        """Calculate POS adoption rate for specific divisions"""
        # Get customers with transactions
        customers_with_pos = set(self.transactions_df['customer_id'].unique())
        
        # Filter customers_df for specific divisions
        filtered_customers = self.customers_df[self.customers_df['division'].isin(divisions)]
        
        # Create a new column indicating if customer has POS
        filtered_customers['has_pos'] = filtered_customers['customer_id'].isin(customers_with_pos)
        
        # Group by division and calculate adoption rate
        adoption_by_division = filtered_customers.groupby('division').agg(
            customers_with_pos=('has_pos', 'sum'),
            total_customers=('customer_id', 'nunique')
        ).reset_index()
        
        # Calculate adoption rate
        adoption_by_division['adoption_rate'] = adoption_by_division['customers_with_pos'] / adoption_by_division['total_customers']
        adoption_by_division['adoption_percentage'] = adoption_by_division['adoption_rate'] * 100
        
        return adoption_by_division
    
    def segment_customers_by_value(self):
        """
        Segment customers by monthly transaction value:
        - High Value: ≥ 7,500,000
        - Medium Value: 3,000,000 - 7,499,999
        - Low Value: ≤ 2,999,999
        """
        # Calculate total transaction value per customer per month
        self.monthly_customer_values = self.transactions_df.groupby(['customer_id', 'month'])['amount'].sum().reset_index()
        
        # Define segmentation function
        def segment_value(amount):
            if amount >= 7_500_000:
                return 'High Value Customer'
            elif 3_000_000 <= amount < 7_500_000:
                return 'Medium Value Customer'
            else:
                return 'Low Value Customer'
        
        # Apply segmentation
        self.monthly_customer_values['segment'] = self.monthly_customer_values['amount'].apply(segment_value)
        
        # Count months per segment for each customer
        segment_counts = self.monthly_customer_values.groupby(['customer_id', 'segment']).size().reset_index(name='months_in_segment')
        
        # Pivot to get counts by segment
        customer_segments = segment_counts.pivot(index='customer_id', columns='segment', values='months_in_segment').fillna(0)
        
        # Calculate total months
        total_months = self.monthly_customer_values['month'].nunique()
        
        # Identify consistently high and low value customers
        self.customer_segments = pd.DataFrame(index=customer_segments.index)
        self.customer_segments['consistently_high_value'] = (customer_segments.get('High Value Customer', pd.Series(0, index=customer_segments.index)) == total_months)
        self.customer_segments['consistently_low_value'] = (customer_segments.get('Low Value Customer', pd.Series(0, index=customer_segments.index)) == total_months)
        
        # Get customer info
        self.customer_segments = pd.merge(
            self.customer_segments.reset_index(),
            self.customers_df[['customer_id', 'subsector_name', 'segment', 'division', 'location']].drop_duplicates('customer_id'),
            on='customer_id'
        )
        
        return self.monthly_customer_values, self.customer_segments
    
    def extract_consistent_customers(self):
        """Extract consistently high and low value customers to CSV"""
        if self.customer_segments is None:
            self.segment_customers_by_value()
        
        # Extract high value customers
        high_value_customers = self.customer_segments[self.customer_segments['consistently_high_value']]
        high_value_path = os.path.join(self.output_dir, 'data', 'high_value_customers.csv')
        high_value_customers.to_csv(high_value_path, index=False)
        
        # Extract low value customers
        low_value_customers = self.customer_segments[self.customer_segments['consistently_low_value']]
        low_value_path = os.path.join(self.output_dir, 'data', 'low_value_customers.csv')
        low_value_customers.to_csv(low_value_path, index=False)
        
        return {
            'high_value_path': high_value_path,
            'low_value_count': len(high_value_customers),
            'low_value_path': low_value_path,
            'low_value_count': len(low_value_customers)
        }
    
    def analyze_value_by_category(self, category):
        """Analyze transaction volume and value by a specific category"""
        if self.merged_df is None:
            self.merge_data()
        
        # Transaction volume (count)
        volume_by_category = self.merged_df.groupby(category).size().reset_index(name='volume')
        
        # Transaction value (sum of amount)
        value_by_category = self.merged_df.groupby(category)['amount'].sum().reset_index()
        
        # Merge volume and value
        result = pd.merge(volume_by_category, value_by_category, on=category)
        
        return result
    
    def get_top_bottom_divisions(self):
        """Get top 10 and bottom 10 divisions by volume and value"""
        division_analysis = self.analyze_value_by_category('division')
        
        # Top 10 by volume
        top_volume = division_analysis.sort_values('volume', ascending=False).head(10)
        
        # Bottom 10 by volume
        bottom_volume = division_analysis.sort_values('volume').head(10)
        
        # Top 10 by value
        top_value = division_analysis.sort_values('amount', ascending=False).head(10)
        
        # Bottom 10 by value
        bottom_value = division_analysis.sort_values('amount').head(10)
        
        return {
            'top_volume': top_volume,
            'bottom_volume': bottom_volume,
            'top_value': top_value,
            'bottom_value': bottom_value
        }
    
    def get_value_segments_by_location(self):
        """Analyze High and Low value customer segments by location"""
        if self.customer_segments is None:
            self.segment_customers_by_value()
        
        # High value customers by location
        high_value_by_location = self.customer_segments[self.customer_segments['consistently_high_value']].groupby('location').size().reset_index(name='high_value_count')
        
        # Low value customers by location
        low_value_by_location = self.customer_segments[self.customer_segments['consistently_low_value']].groupby('location').size().reset_index(name='low_value_count')
        
        # Total customers by location for percentage calculation
        total_by_location = self.customers_df.groupby('location')['customer_id'].nunique().reset_index(name='total_customers')
        
        # Merge high value and total
        high_value_analysis = pd.merge(high_value_by_location, total_by_location, on='location', how='right').fillna(0)
        high_value_analysis['percentage'] = (high_value_analysis['high_value_count'] / high_value_analysis['total_customers']) * 100
        
        # Merge low value and total
        low_value_analysis = pd.merge(low_value_by_location, total_by_location, on='location', how='right').fillna(0)
        low_value_analysis['percentage'] = (low_value_analysis['low_value_count'] / low_value_analysis['total_customers']) * 100
        
        return high_value_analysis, low_value_analysis
    
    def create_matplotlib_visualizations(self):
        """Create static visualizations using Matplotlib and Seaborn"""
        print("Generating static visualizations with Matplotlib and Seaborn...")
        
        # 1. Volume by subsector_name
        subsector_volume = self.analyze_value_by_category('subsector_name')
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='subsector_name', y='volume', data=subsector_volume.sort_values('volume', ascending=False))
        plt.title('Transaction Volume by Subsector', fontsize=16)
        plt.xlabel('Subsector', fontsize=14)
        plt.ylabel('Volume (Count)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add commas to y-axis labels
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'volume_by_subsector.png'), dpi=300)
        plt.close()
        
        # 2. Volume by segment
        segment_volume = self.analyze_value_by_category('segment')
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='segment', y='volume', data=segment_volume.sort_values('volume', ascending=False))
        plt.title('Transaction Volume by Segment', fontsize=16)
        plt.xlabel('Segment', fontsize=14)
        plt.ylabel('Volume (Count)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'volume_by_segment.png'), dpi=300)
        plt.close()
        
        # 3. Volume by division (top 10)
        division_data = self.get_top_bottom_divisions()
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='division', y='volume', data=division_data['top_volume'])
        plt.title('Top 10 Divisions by Transaction Volume', fontsize=16)
        plt.xlabel('Division', fontsize=14)
        plt.ylabel('Volume (Count)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'top10_divisions_volume.png'), dpi=300)
        plt.close()
        
        # 4. Value by subsector_name
        subsector_value = self.analyze_value_by_category('subsector_name')
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='subsector_name', y='amount', data=subsector_value.sort_values('amount', ascending=False))
        plt.title('Transaction Value by Subsector', fontsize=16)
        plt.xlabel('Subsector', fontsize=14)
        plt.ylabel('Value (Amount)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'value_by_subsector.png'), dpi=300)
        plt.close()
        
        # 5. Value by segment
        segment_value = self.analyze_value_by_category('segment')
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='segment', y='amount', data=segment_value.sort_values('amount', ascending=False))
        plt.title('Transaction Value by Segment', fontsize=16)
        plt.xlabel('Segment', fontsize=14)
        plt.ylabel('Value (Amount)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'value_by_segment.png'), dpi=300)
        plt.close()
        
        # 6. Value by division (top 10)
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='division', y='amount', data=division_data['top_value'])
        plt.title('Top 10 Divisions by Transaction Value', fontsize=16)
        plt.xlabel('Division', fontsize=14)
        plt.ylabel('Value (Amount)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'top10_divisions_value.png'), dpi=300)
        plt.close()
        
        # 7. POS Adoption Rate
        adoption_data = self.calculate_pos_adoption()
        plt.figure(figsize=(10, 8))
        labels = ['Customers with POS', 'Customers without POS']
        sizes = [adoption_data['customers_with_pos'], 
                adoption_data['total_customers'] - adoption_data['customers_with_pos']]
        colors = ['#3498db', '#e74c3c']
        explode = (0.1, 0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140, textprops={'fontsize': 14})
        plt.axis('equal')
        plt.title('POS Terminal Adoption Rate', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'pos_adoption_rate.png'), dpi=300)
        plt.close()
        
        # 8. POS Adoption by Location
        adoption_by_location = self.calculate_pos_adoption_by_location()
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='location', y='adoption_percentage', data=adoption_by_location.sort_values('adoption_percentage', ascending=False))
        plt.title('POS Terminal Adoption Rate by Location', fontsize=16)
        plt.xlabel('Location', fontsize=14)
        plt.ylabel('Adoption Rate (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.1f}%', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=10, rotation=0)
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'pos_adoption_by_location.png'), dpi=300)
        plt.close()
        
        # 9. Customer Value Segmentation
        if self.monthly_customer_values is None:
            self.segment_customers_by_value()
        
        segment_counts = self.monthly_customer_values.groupby('segment').size().reset_index(name='count')
        plt.figure(figsize=(10, 8))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax = sns.barplot(x='segment', y='count', data=segment_counts, palette=colors)
        plt.title('Customer Value Segmentation', fontsize=16)
        plt.xlabel('Value Segment', fontsize=14)
        plt.ylabel('Number of Customer-Months', fontsize=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        
        # Add count labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{int(p.get_height()):,}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=12, rotation=0)
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'customer_value_segmentation.png'), dpi=300)
        plt.close()
        
        # 10. High Value Customers by Location
        high_value, low_value = self.get_value_segments_by_location()
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='location', y='percentage', data=high_value.sort_values('percentage', ascending=False))
        plt.title('High Value Customers by Location (% of Total Customers)', fontsize=16)
        plt.xlabel('Location', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.1f}%', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=10, rotation=0)
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'high_value_by_location.png'), dpi=300)
        plt.close()
        
        # 11. Low Value Customers by Location
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='location', y='percentage', data=low_value.sort_values('percentage', ascending=False))
        plt.title('Low Value Customers by Location (% of Total Customers)', fontsize=16)
        plt.xlabel('Location', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.1f}%', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=10, rotation=0)
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'low_value_by_location.png'), dpi=300)
        plt.close()
        
        # 12. POS Adoption in Top/Bottom Divisions
        # Top 10 divisions by volume
        top_divisions = division_data['top_volume']['division'].tolist()
        adoption_top_divisions = self.calculate_pos_adoption_by_division(top_divisions)
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='division', y='adoption_percentage', data=adoption_top_divisions.sort_values('adoption_percentage', ascending=False))
        plt.title('POS Adoption in Top 10 Divisions by Volume', fontsize=16)
        plt.xlabel('Division', fontsize=14)
        plt.ylabel('Adoption Rate (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.1f}%', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=10, rotation=0)
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'pos_adoption_top_divisions.png'), dpi=300)
        plt.close()
        
        print(f"Static visualizations saved to {os.path.join(self.output_dir, 'visualizations')}")
    
    def create_plotly_dashboard(self):
        """Create interactive dashboard using Plotly"""
        print("Generating interactive dashboard with Plotly...")
        
        # Get analysis data
        subsector_volume = self.analyze_value_by_category('subsector_name')
        segment_volume = self.analyze_value_by_category('segment')
        division_volume = self.analyze_value_by_category('division')
        adoption_data = self.calculate_pos_adoption()
        adoption_by_location = self.calculate_pos_adoption_by_location()
        division_data = self.get_top_bottom_divisions()
        high_value, low_value = self.get_value_segments_by_location()
        
        if self.monthly_customer_values is None:
            self.segment_customers_by_value()
        customer_segments = self.monthly_customer_values.groupby('segment').size().reset_index(name='count')
        
        # Create a subplot figure with different sections
        fig = make_subplots(
            rows=6, cols=2,
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar", "colspan": 2}, None],
            ],
            subplot_titles=(
                'Volume by Subsector', 'Value by Subsector',
                'Volume by Segment', 'Value by Segment',
                'POS Terminal Adoption', 'Customer Value Segmentation',
                'Top 10 Divisions by Volume', 'Top 10 Divisions by Value',
                'POS Adoption by Location', 'High Value Customers by Location',
                'POS Adoption in Top 10 Divisions by Volume',
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )
        
        # 1. Volume by subsector_name (row 1, col 1)
        subsector_volume_sorted = subsector_volume.sort_values('volume', ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=subsector_volume_sorted['subsector_name'],
                y=subsector_volume_sorted['volume'],
                text=[f'{int(x):,}' for x in subsector_volume_sorted['volume']],
                textposition='auto',
                marker_color='#3498db',
                name='Volume by Subsector'
            ),
            row=1, col=1
        )
        
        # 2. Value by subsector_name (row 1, col 2)
        subsector_value_sorted = subsector_volume.sort_values('amount', ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=subsector_value_sorted['subsector_name'],
                y=subsector_value_sorted['amount'],
                text=[f'₦{int(x):,}' for x in subsector_value_sorted['amount']],
                textposition='auto',
                marker_color='#2ecc71',
                name='Value by Subsector'
            ),
            row=1, col=2
        )
        
        # 3. Volume by segment (row 2, col 1)
        segment_volume_sorted = segment_volume.sort_values('volume', ascending=False)
        fig.add_trace(
            go.Bar(
                x=segment_volume_sorted['segment'],
                y=segment_volume_sorted['volume'],
                text=[f'{int(x):,}' for x in segment_volume_sorted['volume']],
                textposition='auto',
                marker_color='#3498db',
                name='Volume by Segment'
            ),
            row=2, col=1
        )
        
        # 4. Value by segment (row 2, col 2)
        segment_value_sorted = segment_volume.sort_values('amount', ascending=False)
        fig.add_trace(
            go.Bar(
                x=segment_value_sorted['segment'],
                y=segment_value_sorted['amount'],
                text=[f'₦{int(x):,}' for x in segment_value_sorted['amount']],
                textposition='auto',
                marker_color='#2ecc71',
                name='Value by Segment'
            ),
            row=2, col=2
        )
        
        # 5. POS Terminal Adoption (row 3, col 1)
        fig.add_trace(
            go.Pie(
                labels=['Customers with POS', 'Customers without POS'],
                values=[
                    adoption_data['customers_with_pos'],
                    adoption_data['total_customers'] - adoption_data['customers_with_pos']
                ],
                text=[
                    f"{adoption_data['adoption_percentage']:.1f}%",
                    f"{100 - adoption_data['adoption_percentage']:.1f}%"
                ],
                textinfo='label+percent',
                marker=dict(colors=['#3498db', '#e74c3c']),
                hole=0.4,
                name='POS Adoption'
            ),
            row=3, col=1
        )
        
        # 6. Customer Value Segmentation (row 3, col 2)
        customer_segments_sorted = customer_segments.sort_values('count', ascending=True)
        colors = {'High Value Customer': '#2ecc71', 'Medium Value Customer': '#f39c12', 'Low Value Customer': '#e74c3c'}
        fig.add_trace(
            go.Bar(
                x=customer_segments_sorted['segment'],
                y=customer_segments_sorted['count'],
                text=[f'{int(x):,}' for x in customer_segments_sorted['count']],
                textposition='auto',
                marker_color=[colors[seg] for seg in customer_segments_sorted['segment']],
                name='Customer Segments'
            ),
            row=3, col=2
        )
        
        # 7. Top 10 Divisions by Volume (row 4, col 1)
        fig.add_trace(
            go.Bar(
                x=division_data['top_volume']['division'],
                y=division_data['top_volume']['volume'],
                text=[f'{int(x):,}' for x in division_data['top_volume']['volume']],
                textposition='auto',
                marker_color='#3498db',
                name='Top 10 Divisions by Volume'
            ),
            row=4, col=1
        )
        
        # 8. Top 10 Divisions by Value (row 4, col 2)
        fig.add_trace(
            go.Bar(
                x=division_data['top_value']['division'],
                y=division_data['top_value']['amount'],
                text=[f'₦{int(x):,}' for x in division_data['top_value']['amount']],
                textposition='auto',
                marker_color='#2ecc71',
                name='Top 10 Divisions by Value'
            ),
            row=4, col=2
        )
        
        # 9. POS Adoption by Location (row 5, col 1)
        adoption_by_location_sorted = adoption_by_location.sort_values('adoption_percentage', ascending=False)
        fig.add_trace(
            go.Bar(
                x=adoption_by_location_sorted['location'],
                y=adoption_by_location_sorted['adoption_percentage'],
                text=[f'{x:.1f}%' for x in adoption_by_location_sorted['adoption_percentage']],
                textposition='auto',
                marker_color='#9b59b6',
                name='POS Adoption by Location'
            ),
            row=5, col=1
        )
        
        # 10. High Value Customers by Location (row 5, col 2)
        high_value_sorted = high_value.sort_values('percentage', ascending=False)
        fig.add_trace(
            go.Bar(
                x=high_value_sorted['location'],
                y=high_value_sorted['percentage'],
                text=[f'{x:.1f}%' for x in high_value_sorted['percentage']],
                textposition='auto',
                marker_color='#2ecc71',
                name='High Value Customers by Location'
            ),
            row=5, col=2
        )
        
        # 11. POS Adoption in Top 10 Divisions (row 6, col 1-2)
        top_divisions = division_data['top_volume']['division'].tolist()
        adoption_top_divisions = self.calculate_pos_adoption_by_division(top_divisions)
        adoption_top_sorted = adoption_top_divisions.sort_values('adoption_percentage', ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=adoption_top_sorted['division'],
                y=adoption_top_sorted['adoption_percentage'],
                text=[f'{x:.1f}%' for x in adoption_top_sorted['adoption_percentage']],
                textposition='auto',
                marker_color='#9b59b6',
                name='POS Adoption in Top Divisions'
            ),
            row=6, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text='POS Terminal Analysis Dashboard',
            height=2200,
            width=1200,
            showlegend=False,
            template='plotly_white',
        )
        
        # Update xaxis properties
        for i in range(1, 7):
            for j in range(1, 3):
                if i == 6 and j == 2:
                    continue  # Skip the empty subplot
                fig.update_xaxes(tickangle=45, row=i, col=j)
        
        # Save the interactive dashboard
        dashboard_path = os.path.join(self.output_dir, 'reports', 'interactive_dashboard.html')
        fig.write_html(dashboard_path)
        print(f"Interactive dashboard saved to {dashboard_path}")
        
        return dashboard_path
    
    def generate_insights_report(self):
        """Generate a comprehensive insights report in HTML format"""
        print("Generating comprehensive insights report...")
        
        # Get analysis data
        subsector_volume = self.analyze_value_by_category('subsector_name')
        segment_volume = self.analyze_value_by_category('segment')
        division_volume = self.analyze_value_by_category('division')
        adoption_data = self.calculate_pos_adoption()
        adoption_by_location = self.calculate_pos_adoption_by_location()
        division_data = self.get_top_bottom_divisions()
        high_value, low_value = self.get_value_segments_by_location()
        
        # Calculate additional insights
        top_subsector_volume = subsector_volume.sort_values('volume', ascending=False).iloc[0]
        top_subsector_value = subsector_volume.sort_values('amount', ascending=False).iloc[0]
        
        top_segment_volume = segment_volume.sort_values('volume', ascending=False).iloc[0]
        top_segment_value = segment_volume.sort_values('amount', ascending=False).iloc[0]
        
        top_location_adoption = adoption_by_location.sort_values('adoption_percentage', ascending=False).iloc[0]
        bottom_location_adoption = adoption_by_location.sort_values('adoption_percentage').iloc[0]
        
        # HTML template with insights
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>POS Terminal Analysis - Comprehensive Insights Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    background-color: #f9f9f9;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #fff;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                header {{
                    background: linear-gradient(135deg, #3498db, #2c3e50);
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 40px;
                }}
                h3 {{
                    color: #3498db;
                    margin-top: 25px;
                }}
                .timestamp {{
                    font-style: italic;
                    margin-top: 10px;
                    font-size: 0.9em;
                    opacity: 0.8;
                }}
                .executive-summary {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-left: 4px solid #3498db;
                    margin: 20px 0;
                }}
                .key-metric {{
                    background-color: #f1f8ff;
                    border: 1px solid #c8e1ff;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 15px 0;
                    display: flex;
                    align-items: center;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2c3e50;
                    margin-right: 20px;
                    min-width: 150px;
                    text-align: center;
                }}
                .metric-details {{
                    flex-grow: 1;
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .data-table th, .data-table td {{
                    padding: 12px 15px;
                    border: 1px solid #ddd;
                    text-align: left;
                }}
                .data-table th {{
                    background-color: #3498db;
                    color: white;
                }}
                .data-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .recommendations {{
                    background-color: #ebf5fb;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 25px 0;
                }}
                .recommendation-item {{
                    margin: 15px 0;
                    padding-left: 20px;
                    border-left: 3px solid #3498db;
                }}
                .charts {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin: 30px 0;
                }}
                .chart {{
                    width: 48%;
                    margin-bottom: 20px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.1);
                    padding: 15px;
                    background-color: white;
                    border-radius: 5px;
                }}
                .chart img {{
                    width: 100%;
                    height: auto;
                }}
                footer {{
                    margin-top: 50px;
                    padding: 20px;
                    background-color: #2c3e50;
                    color: white;
                    text-align: center;
                }}
                @media (max-width: 768px) {{
                    .chart {{
                        width: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>POS Terminal Analysis</h1>
                    <p>Comprehensive Insights Report</p>
                    <div class="timestamp">Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </header>
                
                <section class="executive-summary">
                    <h2>Executive Summary</h2>
                    <p>This report provides a detailed analysis of POS terminal transactions across various customer segments, locations, and divisions. The analysis highlights key performance metrics, adoption rates, and customer segmentation to drive strategic business decisions.</p>
                    
                    <div class="key-metric">
                        <div class="metric-value">{adoption_data['adoption_percentage']:.1f}%</div>
                        <div class="metric-details">
                            <h3>Overall POS Adoption Rate</h3>
                            <p>{adoption_data['customers_with_pos']:,} out of {adoption_data['total_customers']:,} customers have POS terminals assigned to them.</p>
                        </div>
                    </div>
                    
                    <div class="key-metric">
                        <div class="metric-value">{top_subsector_volume['subsector_name']}</div>
                        <div class="metric-details">
                            <h3>Top Performing Subsector by Volume</h3>
                            <p>With {int(top_subsector_volume['volume']):,} transactions, this subsector leads in transaction volume.</p>
                        </div>
                    </div>
                    
                    <div class="key-metric">
                        <div class="metric-value">{top_subsector_value['subsector_name']}</div>
                        <div class="metric-details">
                            <h3>Top Performing Subsector by Value</h3>
                            <p>With ₦{int(top_subsector_value['amount']):,} in transactions, this subsector leads in transaction value.</p>
                        </div>
                    </div>
                </section>
                
                <h2>Transaction Analysis</h2>
                
                <h3>Transaction Volume by Category</h3>
                <p>Analysis of transaction counts across different business categories reveals key operational insights about customer activity and engagement.</p>
                
                <div class="charts">
                    <div class="chart">
                        <h4>Volume by Subsector</h4>
                        <img src="../visualizations/volume_by_subsector.png" alt="Volume by Subsector">
                    </div>
                    <div class="chart">
                        <h4>Volume by Segment</h4>
                        <img src="../visualizations/volume_by_segment.png" alt="Volume by Segment">
                    </div>
                </div>
                
                <table class="data-table">
                    <tr>
                        <th>Subsector</th>
                        <th>Transaction Volume</th>
                        <th>% of Total</th>
                    </tr>
                    {
                        ''.join([
                            f"<tr><td>{row['subsector_name']}</td><td>{int(row['volume']):,}</td><td>{row['volume']/subsector_volume['volume'].sum()*100:.1f}%</td></tr>"
                            for _, row in subsector_volume.sort_values('volume', ascending=False).head(5).iterrows()
                        ])
                    }
                </table>
                
                <h3>Transaction Value by Category</h3>
                <p>Analysis of transaction values provides insights into revenue generation across different business categories.</p>
                
                <div class="charts">
                    <div class="chart">
                        <h4>Value by Subsector</h4>
                        <img src="../visualizations/value_by_subsector.png" alt="Value by Subsector">
                    </div>
                    <div class="chart">
                        <h4>Value by Segment</h4>
                        <img src="../visualizations/value_by_segment.png" alt="Value by Segment">
                    </div>
                </div>
                
                <table class="data-table">
                    <tr>
                        <th>Subsector</th>
                        <th>Transaction Value</th>
                        <th>% of Total</th>
                    </tr>
                    {
                        ''.join([
                            f"<tr><td>{row['subsector_name']}</td><td>₦{int(row['amount']):,}</td><td>{row['amount']/subsector_volume['amount'].sum()*100:.1f}%</td></tr>"
                            for _, row in subsector_volume.sort_values('amount', ascending=False).head(5).iterrows()
                        ])
                    }
                </table>
                
                <h2>Division Performance Analysis</h2>
                
                <h3>Top and Bottom Performing Divisions</h3>
                <p>Identifying high and low performing divisions helps prioritize resources and improvement initiatives.</p>
                
                <div class="charts">
                    <div class="chart">
                        <h4>Top 10 Divisions by Volume</h4>
                        <img src="../visualizations/top10_divisions_volume.png" alt="Top 10 Divisions by Volume">
                    </div>
                    <div class="chart">
                        <h4>Top 10 Divisions by Value</h4>
                        <img src="../visualizations/top10_divisions_value.png" alt="Top 10 Divisions by Value">
                    </div>
                </div>
                
                <h3>Top 10 Divisions by Transaction Volume</h3>
                <table class="data-table">
                    <tr>
                        <th>Division</th>
                        <th>Transaction Volume</th>
                        <th>% of Total</th>
                    </tr>
                    {
                        ''.join([
                            f"<tr><td>{row['division']}</td><td>{int(row['volume']):,}</td><td>{row['volume']/division_volume['volume'].sum()*100:.1f}%</td></tr>"
                            for _, row in division_data['top_volume'].iterrows()
                        ])
                    }
                </table>
                
                <h3>Bottom 10 Divisions by Transaction Volume</h3>
                <table class="data-table">
                    <tr>
                        <th>Division</th>
                        <th>Transaction Volume</th>
                        <th>% of Total</th>
                    </tr>
                    {
                        ''.join([
                            f"<tr><td>{row['division']}</td><td>{int(row['volume']):,}</td><td>{row['volume']/division_volume['volume'].sum()*100:.1f}%</td></tr>"
                            for _, row in division_data['bottom_volume'].iterrows()
                        ])
                    }
                </table>
                
                <h2>POS Terminal Adoption Analysis</h2>
                
                <h3>Overall POS Adoption</h3>
                <p>Analysis of POS terminal adoption across the customer base reveals opportunities for expansion.</p>
                
                <div class="charts">
                    <div class="chart">
                        <h4>POS Terminal Adoption Rate</h4>
                        <img src="../visualizations/pos_adoption_rate.png" alt="POS Adoption Rate">
                    </div>
                    <div class="chart">
                        <h4>POS Adoption by Location</h4>
                        <img src="../visualizations/pos_adoption_by_location.png" alt="POS Adoption by Location">
                    </div>
                </div>
                
                <h3>POS Adoption by Location</h3>
                <table class="data-table">
                    <tr>
                        <th>Location</th>
                        <th>Customers with POS</th>
                        <th>Total Customers</th>
                        <th>Adoption Rate</th>
                    </tr>
                    {
                        ''.join([
                            f"<tr><td>{row['location']}</td><td>{int(row['customers_with_pos']):,}</td><td>{int(row['total_customers']):,}</td><td>{row['adoption_percentage']:.1f}%</td></tr>"
                            for _, row in adoption_by_location.sort_values('adoption_percentage', ascending=False).iterrows()
                        ])
                    }
                </table>
                
                <h3>POS Adoption in Top Performing Divisions</h3>
                <p>Analysis of POS adoption in top performing divisions helps identify correlation between POS usage and performance.</p>
                
                <div class="charts">
                    <div class="chart">
                        <h4>POS Adoption in Top Divisions</h4>
                        <img src="../visualizations/pos_adoption_top_divisions.png" alt="POS Adoption in Top Divisions">
                    </div>
                </div>
                
                <h2>Customer Value Segmentation</h2>
                
                <h3>Distribution of Customer Value Segments</h3>
                <p>Segmentation of customers based on monthly transaction value helps identify high-value customers and growth opportunities.</p>
                
                <div class="charts">
                    <div class="chart">
                        <h4>Customer Value Segmentation</h4>
                        <img src="../visualizations/customer_value_segmentation.png" alt="Customer Value Segmentation">
                    </div>
                </div>
                
                <h3>High Value Customers by Location</h3>
                <p>Distribution of high value customers by location helps identify geographical areas with high revenue potential.</p>
                
                <div class="charts">
                    <div class="chart">
                        <h4>High Value Customers by Location</h4>
                        <img src="../visualizations/high_value_by_location.png" alt="High Value Customers by Location">
                    </div>
                    <div class="chart">
                        <h4>Low Value Customers by Location</h4>
                        <img src="../visualizations/low_value_by_location.png" alt="Low Value Customers by Location">
                    </div>
                </div>
                
                <section class="recommendations">
                    <h2>Key Recommendations</h2>
                    
                    <div class="recommendation-item">
                        <h3>Expand POS Coverage in High-Value Locations</h3>
                        <p>Focus on increasing POS terminal adoption in locations with high concentration of high-value customers. {top_location_adoption['location']} leads with {top_location_adoption['adoption_percentage']:.1f}% adoption, while {bottom_location_adoption['location']} has only {bottom_location_adoption['adoption_percentage']:.1f}% adoption.</p>
                    </div>
                    
                    <div class="recommendation-item">
                        <h3>Target Underperforming Divisions</h3>
                        <p>Create targeted strategies for the bottom 10 divisions to improve their transaction volume and value. Consider specialized training, marketing support, or revised commission structures.</p>
                    </div>
                    
                    <div class="recommendation-item">
                        <h3>Optimize Customer Segmentation Strategy</h3>
                        <p>Develop tailored approaches for each customer value segment. Provide premium services to high-value customers, growth incentives for medium-value customers, and activation campaigns for low-value customers.</p>
                    </div>
                    
                    <div class="recommendation-item">
                        <h3>Focus on Top Performing Subsectors</h3>
                        <p>Allocate resources towards expanding market share in the top performing subsectors like {top_subsector_volume['subsector_name']} and {top_subsector_value['subsector_name']} to maximize returns.</p>
                    </div>
                    
                    <div class="recommendation-item">
                        <h3>Implement Retention Programs for High-Value Customers</h3>
                        <p>Develop loyalty programs specifically targeted at retaining high-value customers and increasing their transaction frequency.</p>
                    </div>
                </section>
                
                <footer>
                    <p>This report was automatically generated by the POS Analysis Model.</p>
                    <p>For more detailed analysis, please refer to the interactive dashboard.</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML report
        report_path = os.path.join(self.output_dir, 'reports', 'comprehensive_insights_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comprehensive insights report saved to {report_path}")
        return report_path
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        # Load and prepare data
        if not self.load_data():
            return "Error loading data"
        
        if not self.merge_data():
            return "Error merging data"
        
        # Run analyses
        self.analyze_value_by_category('subsector_name')
        self.analyze_value_by_category('segment')
        self.analyze_value_by_category('division')
        
        self.calculate_pos_adoption()
        self.calculate_pos_adoption_by_location()
        
        self.get_top_bottom_divisions()
        
        self.segment_customers_by_value()
        self.extract_consistent_customers()
        
        # Generate visualizations and reports
        self.create_matplotlib_visualizations()
        self.create_plotly_dashboard()
        self.generate_insights_report()
        
        return "Analysis completed successfully"