    def get_market_data(self, market_id: str) -> Dict[str, Any]:
        """
        Retrieve market data from Kalshi API.
        
        Args:
            market_id: The Kalshi market ID
            
        Returns:
            Dictionary with market information
            
        Raises:
            Exception: If API client is not initialized or request fails
        """    def get_market(self, market_id: str) -> Dict:
        """
        Get details for a specific market.
        
        Args:
            market_id: The Kalshi market ID
            
        Returns:
            Market details
            
        Raises:
            Exception: If the API request fails
        """"""
Simplified Kalshi Prediction Market Analyzer

A streamlined tool for analyzing betting opportunities on Kalshi prediction markets,
focusing only on expected value and optimal bet sizing, with market data retrieval.
"""

import argparse
import json
import logging
import sys
import requests
import datetime
from typing import Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kalshi_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("kalshi_analyzer")


class KalshiMarketAPI:
    """Simple API client for Kalshi markets."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = 'https://demo-api.kalshi.co'):
        """
        Initialize the Kalshi API client.
        
        Args:
            api_key: Your Kalshi API key (optional for public endpoints)
            base_url: Kalshi API base URL (defaults to demo environment)
        """
        self.api_key = api_key
        self.base_url = base_url
        logger.info(f"Initialized Kalshi Market API client with base URL: {base_url}")
    
    def get_markets(self, status: str = "open", limit: int = 10) -> Dict:
        """
        Get a list of markets matching specified criteria.
        
        Args:
            status: Market status filter (open, closed, etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of markets
        """
        # Headers for the request
        headers = {
            "KALSHI-API-KEY": self.api_key
        }
        
        url = f"{self.base_url}/trade-api/v2/markets?status={status}&limit={limit}"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise Exception(f"Failed to retrieve markets: {str(e)}")
        # Headers for the request
        headers = {
            "KALSHI-API-KEY": self.api_key
        }
        
        url = f"{self.base_url}/trade-api/v2/markets/{market_id}"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise Exception(f"Failed to retrieve market data: {str(e)}")


class SimpleKalshiAnalyzer:
    """Simplified analyzer for Kalshi prediction markets."""
    
    def __init__(self, fee_rate: float = 0.05, api_key: Optional[str] = None):
        """
        Initialize the simplified analyzer.
        
        Args:
            fee_rate: Kalshi's transaction fee rate (default: 5%)
            api_key: Kalshi API key for market data retrieval
        """
        self.fee_rate = fee_rate
        self.api = KalshiMarketAPI(api_key=api_key) if api_key else None
        logger.info(f"Initialized SimpleKalshiAnalyzer with fee rate: {fee_rate}")
    
    def analyze_bet(self, your_prob: float, market_prob: float, 
                   bet_type: str = "no") -> Dict[str, Any]:
        """
        Analyze a betting opportunity in a Kalshi market.
        
        Args:
            your_prob: Your estimated probability of the event occurring (0-1)
            market_prob: Market-implied probability of the event occurring (0-1)
            bet_type: Type of bet ("yes" or "no")
            
        Returns:
            Dictionary of simplified analysis results
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not 0 <= your_prob <= 1:
            raise ValueError(f"Your probability must be between 0 and 1, got {your_prob}")
        if not 0 <= market_prob <= 1:
            raise ValueError(f"Market probability must be between 0 and 1, got {market_prob}")
        if bet_type not in ["yes", "no"]:
            raise ValueError(f"Bet type must be 'yes' or 'no', got {bet_type}")
        
        # Calculate edge and expected value
        if bet_type == "yes":
            # For "yes" bets: You win (1/p_market - 1) if event happens, lose 1 if it doesn't
            payout_multiplier = 1 / market_prob
            profit_if_win = payout_multiplier - 1  # Profit per $1 bet
            expected_value = (your_prob * profit_if_win) - ((1 - your_prob) * 1) 
        else:  # "no"
            # For "no" bets: You win (1/(1-p_market) - 1) if event doesn't happen, lose 1 if it does
            payout_multiplier = 1 / (1 - market_prob)
            profit_if_win = payout_multiplier - 1  # Profit per $1 bet
            expected_value = ((1 - your_prob) * profit_if_win) - (your_prob * 1)
        
        # Apply platform fee
        net_expected_value = expected_value * (1 - self.fee_rate)
        
        # Calculate Kelly criterion optimal bet size
        if bet_type == "yes":
            win_prob = your_prob
            net_odds = profit_if_win
        else:
            win_prob = 1 - your_prob
            net_odds = profit_if_win
            
        kelly_fraction = (win_prob * (net_odds + 1) - 1) / net_odds if net_odds > 0 else 0
        kelly_fraction = max(0, min(kelly_fraction, 1))  # Bound between 0 and 1
        
        # Decision recommendation
        decision = "Bet" if net_expected_value > 0 else "Don't Bet"
        edge = abs(your_prob - market_prob)
        
        return {
            "Bet Type": bet_type.capitalize(),
            "Your Probability": your_prob,
            "Market Probability": market_prob,
            "Probability Edge": edge,
            "Payout Multiplier": payout_multiplier,
            "Expected Value (EV)": expected_value,
            "Net EV after Fee": net_expected_value,
            "Kelly Bet Size": kelly_fraction,
            "Decision": decision
        }
    
    def visualize_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Create a simple visualization of the betting analysis.
        
        Args:
            analysis: Results from analyze_bet
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Probability comparison
        labels = ['Your Probability', 'Market Probability']
        probs = [analysis['Your Probability'], analysis['Market Probability']]
        colors = ['blue', 'orange']
        
        ax1.bar(labels, probs, color=colors)
        ax1.set_ylim(0, 1)
        ax1.set_title('Probability Comparison')
        ax1.set_ylabel('Probability')
        for i, v in enumerate(probs):
            ax1.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        # Plot 2: Expected value and Kelly bet size
        metrics = ['Net EV after Fee', 'Kelly Bet Size']
        values = [analysis['Net EV after Fee'], analysis['Kelly Bet Size']]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        ax2.bar(metrics, values, color=colors)
        ax2.set_title('Betting Metrics')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        for i, v in enumerate(values):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        # Add decision as figure title
        decision = analysis['Decision']
        fig.suptitle(f"Decision: {decision}", fontsize=16, 
                    color='green' if decision == "Bet" else 'red')
        
        plt.tight_layout()
        plt.savefig("simple_analysis.png")
        plt.show()
    
    def optimal_bet_analysis(self, your_prob: float, market_prob: float) -> Dict[str, Any]:
        """
        Determine which bet type (yes/no) is optimal for the given probabilities.
        
        Args:
            your_prob: Your estimated probability
            market_prob: Market probability
            
        Returns:
            Analysis of the optimal bet
        """
        yes_analysis = self.analyze_bet(your_prob, market_prob, "yes")
        no_analysis = self.analyze_bet(your_prob, market_prob, "no")
        
        # Determine which has better expected value
        if yes_analysis["Net EV after Fee"] > no_analysis["Net EV after Fee"]:
            return yes_analysis
        else:
            return no_analysis
            
    def list_markets(self, status: str = "open", limit: int = 10) -> Dict[str, Any]:
        """
        List available markets from Kalshi API.
        
        Args:
            status: Filter by market status (open, closed, etc.)
            limit: Maximum number of markets to return
            
        Returns:
            Dictionary with market listings
        """
        if not self.api:
            raise Exception("API client not initialized. Please provide an API key.")
        
        # Get markets from API
        response = self.api.get_markets(status=status, limit=limit)
        
        return {
            "markets": response.get("markets", []),
            "count": len(response.get("markets", [])),
            "status": status
        }
        if not self.api:
            raise Exception("API client not initialized. Please provide an API key.")
        
        # Get market data from API
        market_data = self.api.get_market(market_id)
        
        # Extract key information
        market = market_data.get('market', {})
        
        # Get current prices
        yes_price = float(market.get('yes_bid', 0))
        no_price = 1 - float(market.get('yes_ask', 1))
        
        # Calculate implied probability
        implied_prob = yes_price
        
        # Calculate days to resolution if settlement date is available
        days_to_resolve = None
        if market.get('settlement_date'):
            try:
                settlement_date = datetime.datetime.fromisoformat(
                    market.get('settlement_date').replace('Z', '+00:00')
                )
                now = datetime.datetime.now(datetime.timezone.utc)
                days_to_resolve = (settlement_date - now).days
            except (ValueError, TypeError):
                logger.warning("Could not parse settlement date")
        
        # Return formatted market data
        return {
            "market_id": market.get('id'),
            "title": market.get('title'),
            "subtitle": market.get('subtitle'),
            "yes_price": yes_price,
            "no_price": no_price,
            "implied_probability": implied_prob,
            "days_to_resolve": days_to_resolve,
            "close_date": market.get('close_date'),
            "settlement_date": market.get('settlement_date'),
            "raw_data": market
        }


class SimpleKalshiCLI:
    """Simple command line interface for the Kalshi Analyzer."""
    
    def __init__(self):
        """Initialize the simplified CLI."""
        self.parser = self._setup_argparse()
        
    @staticmethod
    def _setup_argparse() -> argparse.ArgumentParser:
        """
        Set up simplified command line argument parser.
        
        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="Simplified Kalshi Prediction Market Analyzer",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Add subparsers for different commands
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Analyze command
        analyze_parser = subparsers.add_parser("analyze", help="Analyze a bet opportunity")
        analyze_parser.add_argument("--your-prob", type=float, required=True, 
                                  help="Your estimated probability (0-1)")
        analyze_parser.add_argument("--market-prob", type=float, required=True, 
                                  help="Market probability (0-1)")
        analyze_parser.add_argument("--bet-type", choices=["yes", "no"], default="no", 
                                  help="Type of bet to analyze")
        analyze_parser.add_argument("--visualize", action="store_true", 
                                  help="Generate visualization of results")
        
        # Find optimal bet command
        optimal_parser = subparsers.add_parser("optimal", help="Find the optimal bet type")
        optimal_parser.add_argument("--your-prob", type=float, required=True, 
                                  help="Your estimated probability (0-1)")
        optimal_parser.add_argument("--market-prob", type=float, required=True, 
                                  help="Market probability (0-1)")
        optimal_parser.add_argument("--visualize", action="store_true", 
                                  help="Generate visualization of results")
        
        # Market lookup command
        market_parser = subparsers.add_parser("market", help="Get current market prices")
        market_parser.add_argument("--market-id", type=str, required=True, 
                                 help="Kalshi market ID to look up")
        market_parser.add_argument("--analyze", action="store_true",
                                 help="Analyze the market after retrieving data")
        market_parser.add_argument("--your-prob", type=float,
                                 help="Your probability estimate (required if --analyze)")
        
        # List markets command
        list_parser = subparsers.add_parser("list", help="List available markets")
        list_parser.add_argument("--status", type=str, default="open",
                               choices=["open", "closed", "settled", "upcoming"],
                               help="Filter markets by status")
        list_parser.add_argument("--limit", type=int, default=10,
                               help="Maximum number of markets to display")
        
        # Global options
        parser.add_argument("--config", type=str, default="config.json", 
                          help="Path to configuration file")
        parser.add_argument("--api-key", type=str, 
                          help="Kalshi API key (overrides config)")
        parser.add_argument("--fee-rate", type=float, default=0.05, 
                          help="Kalshi fee rate (default: 0.05)")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        
        return parser
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.debug(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {str(e)}")
            return {}
    
    def run(self) -> None:
        """Run the simplified CLI application."""
        args = self.parse_args()
        
        # Set up logging level
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
        # Load configuration
        config = self.load_config(args.config)
        
        # Get API key (priority: CLI args > config file)
        api_key = args.api_key or config.get("api_credentials", {}).get("api_key")
        
        # Get fee rate (priority: CLI args > config file)
        fee_rate = args.fee_rate
        if fee_rate is None:
            fee_rate = config.get("platform_settings", {}).get("fee_rate", 0.05)
        
        # Initialize analyzer
        analyzer = SimpleKalshiAnalyzer(fee_rate=fee_rate, api_key=api_key)
        
        # Execute the selected command
        if args.command == "analyze":
            result = analyzer.analyze_bet(
                your_prob=args.your_prob,
                market_prob=args.market_prob,
                bet_type=args.bet_type
            )
            
            # Display results
            print("\n===== Bet Analysis =====")
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
            # Generate visualization if requested
            if args.visualize:
                analyzer.visualize_analysis(result)
        
        elif args.command == "optimal":
            result = analyzer.optimal_bet_analysis(
                your_prob=args.your_prob,
                market_prob=args.market_prob
            )
            
            # Display results
            print("\n===== Optimal Bet Analysis =====")
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
            # Generate visualization if requested
            if args.visualize:
                analyzer.visualize_analysis(result)
        
        elif args.command == "market":
            # Ensure we have API key
            if not api_key:
                print("Error: API key is required for market lookup.")
                print("Provide it with --api-key or in your config file.")
                sys.exit(1)
            
            try:
                # Get market data
                market_data = analyzer.get_market_data(args.market_id)
                
                # Display market information
                print(f"\n===== Market: {market_data['title']} =====")
                print(f"ID: {market_data['market_id']}")
                if market_data.get('subtitle'):
                    print(f"Subtitle: {market_data['subtitle']}")
                
                # Display pricing information
                print(f"\nCurrent Pricing:")
                print(f"Yes Price: {market_data['yes_price']:.4f}")
                print(f"No Price: {market_data['no_price']:.4f}")
                print(f"Implied Probability: {market_data['implied_probability']:.2%}")
                
                # Display dates if available
                if market_data.get('days_to_resolve'):
                    print(f"\nDays to Resolution: {market_data['days_to_resolve']}")
                if market_data.get('close_date'):
                    print(f"Close Date: {market_data['close_date']}")
                if market_data.get('settlement_date'):
                    print(f"Settlement Date: {market_data['settlement_date']}")
                
                # If analysis requested, perform analysis
                if args.analyze:
                    if args.your_prob is None:
                        print("\nError: --your-prob is required for analysis")
                    else:
                        market_prob = market_data['implied_probability']
                        print(f"\nUsing market probability: {market_prob:.4f}")
                        
                        # Run optimal bet analysis
                        result = analyzer.optimal_bet_analysis(
                            your_prob=args.your_prob,
                            market_prob=market_prob
                        )
                        
                        # Display results
                        print("\n===== Optimal Bet Analysis =====")
                        for key, value in result.items():
                            if isinstance(value, float):
                                print(f"{key}: {value:.4f}")
                            else:
                                print(f"{key}: {value}")
                
                # Suggest next steps
                print("\nSuggested Next Steps:")
                if market_data.get('implied_probability'):
                    mp = market_data['implied_probability']
                    print(f"  analyze --your-prob 0.XX --market-prob {mp:.4f} --bet-type [yes|no]")
                
            except Exception as e:
                logger.error(f"Error retrieving market data: {str(e)}")
                print(f"Error: {str(e)}")
                sys.exit(1)
        
        elif args.command == "list":
            # Ensure we have API key
            if not api_key:
                print("Error: API key is required for listing markets.")
                print("Provide it with --api-key or in your config file.")
                sys.exit(1)
            
            try:
                # Get list of markets
                market_list = analyzer.list_markets(
                    status=args.status,
                    limit=args.limit
                )
                
                # Display market list
                print(f"\n===== Available {args.status.capitalize()} Markets =====")
                print(f"Found {market_list['count']} markets\n")
                
                # Format the list for display
                for market in market_list['markets']:
                    market_id = market.get('id', 'Unknown')
                    title = market.get('title', 'Untitled Market')
                    
                    # Get prices if available
                    yes_bid = market.get('yes_bid')
                    yes_ask = market.get('yes_ask')
                    if yes_bid is not None and yes_ask is not None:
                        price_info = f"Yes: {yes_bid:.2f} | No: {1-yes_ask:.2f}"
                    else:
                        price_info = "No price data"
                    
                    print(f"ID: {market_id}")
                    print(f"Title: {title}")
                    print(f"Prices: {price_info}")
                    print("-" * 50)
                
                # Suggest next steps
                print("\nTo view details for a specific market:")
                print(f"  market --market-id MARKET_ID")
                
            except Exception as e:
                logger.error(f"Error listing markets: {str(e)}")
                print(f"Error: {str(e)}")
                sys.exit(1)
    
    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        return self.parser.parse_args()


def main():
    """Main entry point for the simplified application."""
    try:
        cli = SimpleKalshiCLI()
        cli.run()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.exception("Detailed traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
