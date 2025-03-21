"""
Simplified Kalshi Prediction Market Analyzer

A streamlined tool for analyzing betting opportunities on Kalshi prediction markets,
focusing only on expected value and optimal bet sizing.
"""

import argparse
import json
import logging
import sys
from typing import Dict, Any

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


class SimpleKalshiAnalyzer:
    """Simplified analyzer for Kalshi prediction markets."""
    
    def __init__(self, fee_rate: float = 0.05):
        """
        Initialize the simplified analyzer.
        
        Args:
            fee_rate: Kalshi's transaction fee rate (default: 5%)
        """
        self.fee_rate = fee_rate
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
        
        # Global options
        parser.add_argument("--fee-rate", type=float, default=0.05, 
                          help="Kalshi fee rate (default: 0.05)")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        
        return parser
    
    def run(self) -> None:
        """Run the simplified CLI application."""
        args = self.parse_args()
        
        # Set up logging level
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
        # Initialize analyzer
        analyzer = SimpleKalshiAnalyzer(fee_rate=args.fee_rate)
        
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
        
        else:
            self.parser.print_help()
    
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
